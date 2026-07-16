"""Full fixed-point Spikingformer training runtime for the NeuraEdge fabric.

The processing system owns graph scheduling and DDR tensors. Matrix derivatives,
surrogate-gradient recurrence, pooling scatter, vector arithmetic, softmax, and
SGDW updates are submitted to programmable-logic engines through the command ABI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from neuraedge_compiler import compile_spike_gemm, compile_u8_gemm
from neuraedge_golden import VectorOperation, round_shift_signed
from neuraedge_protocol import (
    Opcode,
    ProgramRecord,
    clear_status_command,
    log_softmax_command,
    pack_softmax_beats,
    pack_vector_beats,
    unpack_softmax_result,
    unpack_vector_result,
    vector_command,
)
from neuraedge_training import (
    TrainingProgramExecutor,
    compile_affine_backward,
    compile_conv2d_backward,
    compile_linear_backward,
    execute_lif_bptt,
    execute_maxpool2d_backward,
    execute_sgdw_update,
)
from spikingformer_accelerator import (
    SpikingformerInferenceResult,
    SpikingformerProjectionExecutor,
    binary_max_pool2d_on_accelerator,
    lif_sequence_with_trace_on_accelerator,
)
from spikingformer_parameters import (
    HeadParameters,
    ProjectionParameters,
    SpikingformerParameterArchive,
)


VECTOR_LANES = 8
MASTER_FORMAT = "neuraedge-spikingformer-trainable-state-v1"
CHECKPOINT_FORMAT = "neuraedge-spikingformer-training-checkpoint-v1"


class FullTrainingDriver(TrainingProgramExecutor, Protocol):
    """Command interface required by both forward inference and backward update."""


@dataclass(frozen=True)
class BlockGradient:
    """Signed integer gradient plus its exact shared power-of-two exponent."""

    values: np.ndarray
    exponent: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.int64)
        if values.size and (
            values.min() < -(1 << 31) or values.max() > (1 << 31) - 1
        ):
            raise OverflowError("block-gradient lanes must fit signed 32-bit")
        object.__setattr__(self, "values", values)
        if not -127 <= int(self.exponent) <= 127:
            raise ValueError("block-gradient exponent must be in [-127,127]")

    def reshape(self, *shape: int) -> "BlockGradient":
        return BlockGradient(self.values.reshape(*shape), self.exponent)

    def transpose(self, *axes: int) -> "BlockGradient":
        return BlockGradient(self.values.transpose(*axes), self.exponent)


@dataclass(frozen=True)
class ParameterGradient:
    """Weight and bias derivatives emitted by one learned affine operator."""

    weight: BlockGradient
    bias: BlockGradient


@dataclass
class MasterTensor:
    """Signed-16 master value, signed-24 Q15 momentum, and deployment format."""

    master: np.ndarray
    velocity_q15: np.ndarray
    fractional_bits: int
    deploy_minimum: int
    deploy_maximum: int

    def __post_init__(self) -> None:
        master = np.asarray(self.master)
        velocity = np.asarray(self.velocity_q15)
        if not np.issubdtype(master.dtype, np.integer) or not np.issubdtype(
            velocity.dtype, np.integer
        ):
            raise ValueError("master and velocity tensors must contain integers")
        if master.shape != velocity.shape or master.size == 0:
            raise ValueError("master and velocity tensors must share a nonempty shape")
        if np.any(master < -(1 << 15)) or np.any(master > (1 << 15) - 1):
            raise ValueError("master tensor must fit signed 16-bit")
        if np.any(velocity < -(1 << 23)) or np.any(velocity > (1 << 23) - 1):
            raise ValueError("master velocity must fit signed 24-bit")
        self.master = master.astype(np.int16, copy=False)
        self.velocity_q15 = velocity.astype(np.int32, copy=False)
        if not 0 <= self.fractional_bits <= 14:
            raise ValueError("master fractional bits must be in [0,14]")
        if self.deploy_minimum > self.deploy_maximum:
            raise ValueError("deployment clamp is inverted")


def _padded_lanes(values: Sequence[int], fill: int = 0) -> tuple[int, ...]:
    lanes = tuple(map(int, values))
    if len(lanes) > VECTOR_LANES:
        raise ValueError("a vector chunk cannot exceed eight lanes")
    return lanes + (int(fill),) * (VECTOR_LANES - len(lanes))


def execute_vector_tensor(
    driver: FullTrainingDriver,
    operation: VectorOperation,
    operand_a,
    operand_b=None,
    *,
    multiplier_q15: int = 0,
    right_shift: int = 0,
    output_zero_point: int = 0,
    threshold: int = 0,
    surrogate_window: int = 0,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Apply one vector command to every element using descriptor batching."""

    a = np.asarray(operand_a, dtype=np.int64)
    b = np.zeros_like(a) if operand_b is None else np.asarray(operand_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("vector operands must share a shape")
    for name, value in (("operand_a", a), ("operand_b", b)):
        if value.size and (
            value.min() < -(1 << 31) or value.max() > (1 << 31) - 1
        ):
            raise OverflowError(f"{name} does not fit signed 32-bit lanes")
    if a.size == 0:
        return np.empty(a.shape, dtype=np.int64)
    command = vector_command(
        int(operation),
        multiplier_q15=multiplier_q15,
        right_shift=right_shift,
        output_zero_point=output_zero_point,
        threshold=threshold,
        surrogate_window=surrogate_window,
    )
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    records: list[ProgramRecord] = []
    counts: list[int] = []
    for start in range(0, flat_a.size, VECTOR_LANES):
        stop = min(start + VECTOR_LANES, flat_a.size)
        count = stop - start
        records.append(ProgramRecord(
            command,
            pack_vector_beats(
                _padded_lanes(flat_a[start:stop]),
                _padded_lanes(flat_b[start:stop]),
                clamp_minimum=clamp_minimum,
                clamp_maximum=clamp_maximum,
            ),
            4,
        ))
        counts.append(count)
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != 4 * len(records):
        raise RuntimeError(
            f"vector program returned {len(words)} beats; expected {4 * len(records)}"
        )
    result = np.empty(flat_a.size, dtype=np.int64)
    cursor = 0
    for record_index, count in enumerate(counts):
        lanes = unpack_vector_result(words[4 * record_index:4 * record_index + 4])
        result[cursor:cursor + count] = lanes[:count]
        cursor += count
    return result.reshape(a.shape)


def execute_power_of_two_scale(
    driver: FullTrainingDriver,
    values,
    power: int,
    *,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Multiply by ``2**power`` with exact fabric round-to-nearest shifts."""

    result = np.asarray(values, dtype=np.int64)
    remaining = int(power)
    while remaining:
        step = min(remaining, 14) if remaining > 0 else max(remaining, -16)
        result = execute_vector_tensor(
            driver,
            VectorOperation.FLEX_REQUANTIZE,
            result,
            multiplier_q15=1 << 14,
            right_shift=-step - 1,
            clamp_minimum=clamp_minimum,
            clamp_maximum=clamp_maximum,
            timeout_seconds=timeout_seconds,
        )
        remaining -= step
    return result


def normalize_gradient_s16(
    driver: FullTrainingDriver,
    gradient: BlockGradient,
    *,
    timeout_seconds: float = 30.0,
) -> BlockGradient:
    """Right-shift a block only as much as required for signed-16 matrix input."""

    values = gradient.values
    maximum = int(np.abs(values).max(initial=0))
    shift = 0
    while maximum > 32767:
        maximum = (maximum + 1) // 2
        shift += 1
    if shift == 0:
        return BlockGradient(values.astype(np.int64, copy=False), gradient.exponent)
    shifted = execute_power_of_two_scale(
        driver,
        values,
        -shift,
        clamp_minimum=-32768,
        clamp_maximum=32767,
        timeout_seconds=timeout_seconds,
    )
    return BlockGradient(shifted, gradient.exponent + shift)


def add_block_gradients(
    driver: FullTrainingDriver,
    left: BlockGradient,
    right: BlockGradient,
    *,
    timeout_seconds: float = 30.0,
) -> BlockGradient:
    """Align two block exponents and add all lanes in programmable logic."""

    if left.values.shape != right.values.shape:
        raise ValueError("gradient addends must share a shape")
    exponent = max(left.exponent, right.exponent)
    left_values = left.values
    right_values = right.values
    if left.exponent != exponent:
        left_values = execute_power_of_two_scale(
            driver, left_values, left.exponent - exponent,
            timeout_seconds=timeout_seconds)
    if right.exponent != exponent:
        right_values = execute_power_of_two_scale(
            driver, right_values, right.exponent - exponent,
            timeout_seconds=timeout_seconds)
    result = execute_vector_tensor(
        driver,
        VectorOperation.ADD_SAT,
        left_values,
        right_values,
        timeout_seconds=timeout_seconds,
    )
    return BlockGradient(result, exponent)


def _required_right_shift(bound: int, maximum: int) -> int:
    if bound < 0 or maximum <= 0:
        raise ValueError("gradient bounds must be nonnegative with a positive limit")
    shift = 0
    while bound > maximum:
        bound = (bound + 1) // 2
        shift += 1
    if shift > 31:
        raise OverflowError("a gradient reduction requires more than 31 shift bits")
    return shift


def backward_requantize_projection(
    driver: FullTrainingDriver,
    output_gradient: BlockGradient,
    parameters: ProjectionParameters,
    *,
    timeout_seconds: float = 30.0,
) -> BlockGradient:
    """Differentiate the exported per-channel folded-BN requantizer with STE."""

    values = output_gradient.values
    channels = parameters.weight_k_by_n.shape[1]
    if values.ndim != 2 or values.shape[1] != channels:
        raise ValueError("projection output gradient must have shape [rows,channels]")
    shifts = parameters.binary_point_shift.astype(np.int64)
    multipliers = parameters.multiplier_q15.astype(np.int64)
    boost_minimum = max(int(shift) - 15 for shift in shifts)
    boost_maximum = min(int(shift) + 15 for shift in shifts)
    selected_boost: int | None = None
    for boost in range(boost_maximum, boost_minimum - 1, -1):
        represented_maximum = 0
        for channel in range(channels):
            product = values[:, channel] * int(multipliers[channel])
            represented = round_shift_signed(
                product, 15 + int(shifts[channel]) - boost)
            represented_maximum = max(
                represented_maximum, int(np.abs(represented).max(initial=0)))
        if represented_maximum <= 32767:
            selected_boost = boost
            break
    if selected_boost is None:
        raise OverflowError("projection derivative cannot be represented by signed-16 lanes")

    rows = values.shape[0]
    records: list[ProgramRecord] = []
    bindings: list[tuple[int, int, int]] = []
    zeros = (0,) * VECTOR_LANES
    for channel in range(channels):
        command = vector_command(
            int(VectorOperation.FLEX_REQUANTIZE),
            multiplier_q15=int(multipliers[channel]),
            right_shift=int(shifts[channel]) - selected_boost,
        )
        for row_start in range(0, rows, VECTOR_LANES):
            row_stop = min(row_start + VECTOR_LANES, rows)
            records.append(ProgramRecord(
                command,
                pack_vector_beats(
                    _padded_lanes(values[row_start:row_stop, channel]),
                    zeros,
                    clamp_minimum=-32768,
                    clamp_maximum=32767,
                ),
                4,
            ))
            bindings.append((row_start, row_stop, channel))
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != 4 * len(records):
        raise RuntimeError("projection derivative returned an invalid stream length")
    result = np.zeros(values.shape, dtype=np.int64)
    for record_index, (row_start, row_stop, channel) in enumerate(bindings):
        lanes = unpack_vector_result(words[4 * record_index:4 * record_index + 4])
        result[row_start:row_stop, channel] = lanes[:row_stop - row_start]
    return BlockGradient(result, output_gradient.exponent - selected_boost)


def execute_linear_backward_block(
    driver: FullTrainingDriver,
    activations,
    weights,
    output_gradient: BlockGradient,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[BlockGradient, BlockGradient]:
    """Run a block-floating matrix backward pass with proven saturation bounds."""

    x = np.asarray(activations, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("linear backward operands do not align")
    if output_gradient.values.shape != (x.shape[0], w.shape[1]):
        raise ValueError("linear output-gradient shape does not match Y=XW")
    dy = normalize_gradient_s16(
        driver, output_gradient, timeout_seconds=timeout_seconds)
    maximum_dy = int(np.abs(dy.values).max(initial=0))
    maximum_x = int(np.abs(x).max(initial=0))
    maximum_w = int(np.abs(w).max(initial=0))
    dx_bound = w.shape[1] * maximum_dy * maximum_w
    dw_bound = x.shape[0] * maximum_x * maximum_dy
    dx_shift = _required_right_shift(dx_bound, (1 << 31) - 1)
    dw_shift = _required_right_shift(dw_bound, (1 << 23) - 1)
    program = compile_linear_backward(
        x,
        w,
        dy.values,
        input_gradient_shift=dx_shift,
        weight_gradient_shift=dw_shift,
    )
    dx, dw = program.execute_batched(driver, timeout_seconds=timeout_seconds)
    return (
        BlockGradient(dx, dy.exponent + dx_shift),
        BlockGradient(dw, dy.exponent + dw_shift),
    )


def execute_affine_backward_block(
    driver: FullTrainingDriver,
    activations,
    weights,
    bias,
    output_gradient: BlockGradient,
    *,
    timeout_seconds: float = 120.0,
) -> tuple[BlockGradient, ParameterGradient]:
    """Run dX, dW, and dB for Y=XW+b through the matrix backward array."""

    x = np.asarray(activations, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    b = np.asarray(bias, dtype=np.int64)
    if x.ndim != 2 or w.ndim != 2 or b.shape != (w.shape[1],):
        raise ValueError("affine operands have inconsistent shapes")
    if x.shape[1] != w.shape[0]:
        raise ValueError("affine activation and weight dimensions do not align")
    dy = normalize_gradient_s16(
        driver, output_gradient, timeout_seconds=timeout_seconds)
    maximum_dy = int(np.abs(dy.values).max(initial=0))
    maximum_x = max(int(np.abs(x).max(initial=0)), 1)
    maximum_w = max(int(np.abs(w).max(initial=0)), int(np.abs(b).max(initial=0)))
    dx_bound = w.shape[1] * maximum_dy * maximum_w
    dw_bound = x.shape[0] * maximum_x * maximum_dy
    dx_shift = _required_right_shift(dx_bound, (1 << 31) - 1)
    dw_shift = _required_right_shift(dw_bound, (1 << 23) - 1)
    program = compile_affine_backward(
        x,
        w,
        b,
        dy.values,
        input_gradient_shift=dx_shift,
        weight_gradient_shift=dw_shift,
    )
    dx, dw, db = program.execute_batched(driver, timeout_seconds=timeout_seconds)
    return (
        BlockGradient(dx, dy.exponent + dx_shift),
        ParameterGradient(
            BlockGradient(dw, dy.exponent + dw_shift),
            BlockGradient(db, dy.exponent + dw_shift),
        ),
    )


def execute_conv2d_backward_block(
    driver: FullTrainingDriver,
    activations,
    weights,
    bias,
    output_gradient: BlockGradient,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    timeout_seconds: float = 120.0,
) -> tuple[BlockGradient, ParameterGradient]:
    """Run grouped NCHW convolution dX/dW/dB with block-floating bounds."""

    x = np.asarray(activations, dtype=np.int64)
    w = np.asarray(weights, dtype=np.int64)
    b = np.asarray(bias, dtype=np.int64)
    dy = output_gradient.values
    if x.ndim != 4 or w.ndim != 4 or dy.ndim != 4:
        raise ValueError("convolution backward requires NCHW/OIHW tensors")
    normalized = normalize_gradient_s16(
        driver, output_gradient, timeout_seconds=timeout_seconds)
    maximum_dy = int(np.abs(normalized.values).max(initial=0))
    maximum_x = max(int(np.abs(x).max(initial=0)), 1)
    maximum_w = int(np.abs(w).max(initial=0))
    output_channels_per_group = w.shape[0] // groups
    kernel_elements = w.shape[2] * w.shape[3]
    dx_bound = output_channels_per_group * kernel_elements * maximum_dy * maximum_w
    reduction_rows = normalized.values.shape[0] * normalized.values.shape[2] * normalized.values.shape[3]
    dw_bound = reduction_rows * maximum_x * maximum_dy
    dx_shift = _required_right_shift(dx_bound, (1 << 31) - 1)
    dw_shift = _required_right_shift(dw_bound, (1 << 23) - 1)
    program = compile_conv2d_backward(
        x,
        w,
        b,
        normalized.values,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        input_gradient_shift=dx_shift,
        weight_gradient_shift=dw_shift,
    )
    dx, dw, db = program.execute_batched(driver, timeout_seconds=timeout_seconds)
    return (
        BlockGradient(dx, normalized.exponent + dx_shift),
        ParameterGradient(
            BlockGradient(dw, normalized.exponent + dw_shift),
            BlockGradient(db, normalized.exponent + dw_shift),
        ),
    )


def execute_softmax_q15(
    driver: FullTrainingDriver,
    logits,
    *,
    difference_shift: int,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Evaluate the hardware power-of-two softmax for each classifier row."""

    scores = np.asarray(logits, dtype=np.int64)
    if scores.ndim != 2 or not 1 <= scores.shape[1] <= 256:
        raise ValueError("logits must have shape [batch,classes] with 1..256 classes")
    if scores.size and (scores.min() < -(1 << 23) or scores.max() > (1 << 23) - 1):
        raise OverflowError("softmax logits must fit signed 24-bit")
    records = tuple(
        ProgramRecord(
            log_softmax_command(scores.shape[1], difference_shift),
            pack_softmax_beats(tuple(map(int, row))),
            scores.shape[1],
        )
        for row in scores
    )
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != scores.size:
        raise RuntimeError("softmax returned an invalid stream length")
    probabilities = np.zeros(scores.shape, dtype=np.int64)
    for batch_index in range(scores.shape[0]):
        for class_index in range(scores.shape[1]):
            probability, returned_index, final = unpack_softmax_result(
                words[batch_index * scores.shape[1] + class_index]
            )
            if returned_index != class_index or final != (
                class_index == scores.shape[1] - 1
            ):
                raise RuntimeError("softmax result metadata is out of order")
            probabilities[batch_index, class_index] = probability
    return probabilities


def execute_cross_entropy_gradient_q15(
    driver: FullTrainingDriver,
    probabilities_q15,
    labels,
    *,
    timeout_seconds: float = 30.0,
) -> BlockGradient:
    """Form batch-mean ``probability - one_hot`` entirely with vector RTL."""

    probabilities = np.asarray(probabilities_q15, dtype=np.int64)
    targets = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or targets.shape != (probabilities.shape[0],):
        raise ValueError("labels must contain one class index per probability row")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError("a target class is outside the classifier range")
    correction = np.zeros(probabilities.shape, dtype=np.int64)
    correction[np.arange(probabilities.shape[0]), targets] = -(1 << 15)
    gradient = execute_vector_tensor(
        driver,
        VectorOperation.ADD_SAT,
        probabilities,
        correction,
        timeout_seconds=timeout_seconds,
    )
    if probabilities.shape[0] > 1:
        reciprocal_q15 = int(round((1 << 15) / probabilities.shape[0]))
        gradient = execute_vector_tensor(
            driver,
            VectorOperation.MULTIPLY_Q15,
            gradient,
            np.full(gradient.shape, reciprocal_q15, dtype=np.int64),
            timeout_seconds=timeout_seconds,
        )
    return BlockGradient(gradient, -15)


def _master_fractional_bits(values: np.ndarray, maximum_bits: int) -> int:
    maximum = int(np.abs(np.asarray(values, dtype=np.int64)).max(initial=0))
    fractional = 0
    while fractional < maximum_bits and maximum <= (32767 >> (fractional + 1)):
        fractional += 1
    return fractional


def execute_deploy_master(
    driver: FullTrainingDriver,
    tensor: MasterTensor,
    *,
    timeout_seconds: float = 120.0,
) -> np.ndarray:
    """Round a signed-16 master tensor and clamp it to its deployment range."""

    shifted = execute_power_of_two_scale(
        driver,
        tensor.master,
        -tensor.fractional_bits,
        clamp_minimum=tensor.deploy_minimum,
        clamp_maximum=tensor.deploy_maximum,
        timeout_seconds=timeout_seconds,
    ) if tensor.fractional_bits else execute_vector_tensor(
        driver,
        VectorOperation.CLAMP,
        tensor.master,
        clamp_minimum=tensor.deploy_minimum,
        clamp_maximum=tensor.deploy_maximum,
        timeout_seconds=timeout_seconds,
    )
    return shifted.astype(np.int64)


def gradient_to_optimizer_q15(
    driver: FullTrainingDriver,
    gradient: BlockGradient,
    *,
    clip_q15: int,
    timeout_seconds: float = 120.0,
) -> np.ndarray:
    """Convert a block gradient to Q15 and apply explicit fabric clipping."""

    if not 1 <= clip_q15 <= (1 << 23) - 1:
        raise ValueError("clip_q15 must be in [1,8388607]")
    scaled = execute_power_of_two_scale(
        driver,
        gradient.values,
        gradient.exponent + 15,
        timeout_seconds=timeout_seconds,
    ) if gradient.exponent != -15 else gradient.values
    clipped = execute_vector_tensor(
        driver,
        VectorOperation.CLAMP,
        scaled,
        clamp_minimum=-clip_q15,
        clamp_maximum=clip_q15,
        timeout_seconds=timeout_seconds,
    )
    return clipped.astype(np.int64)


class TrainableSpikingformerState:
    """Mutable master parameters implementing the read-only archive interface."""

    def __init__(self, archive: SpikingformerParameterArchive) -> None:
        self.manifest = archive.manifest
        self.base_binary_sha256 = str(archive.manifest["binary_sha256"])
        self._projection_static: dict[str, ProjectionParameters] = {}
        self._projection_weight: dict[str, MasterTensor] = {}
        self._projection_bias: dict[str, MasterTensor] = {}
        self._deployed_projection_weight: dict[str, np.ndarray] = {}
        self._deployed_projection_bias: dict[str, np.ndarray] = {}
        for name in archive.projection_names:
            parameters = archive.projection(name)
            weight = parameters.weight_k_by_n.astype(np.int64)
            bias = parameters.bias_q10.astype(np.int64)
            weight_fractional_bits = 8
            bias_fractional_bits = _master_fractional_bits(bias, 8)
            self._projection_static[name] = parameters
            self._projection_weight[name] = MasterTensor(
                weight << weight_fractional_bits,
                np.zeros(weight.shape, dtype=np.int64),
                weight_fractional_bits,
                -128,
                127,
            )
            self._projection_bias[name] = MasterTensor(
                bias << bias_fractional_bits,
                np.zeros(bias.shape, dtype=np.int64),
                bias_fractional_bits,
                -(1 << 15),
                (1 << 15) - 1,
            )
            self._deployed_projection_weight[name] = weight.astype(np.int8)
            self._deployed_projection_bias[name] = bias.astype(np.int64)

        head = archive.head
        head_weight = head.weight_k_by_n.astype(np.int64)
        head_bias = head.bias_accumulator_units.astype(np.int64)
        head_bias_fractional_bits = _master_fractional_bits(head_bias, 8)
        self._head_weight = MasterTensor(
            head_weight << 8,
            np.zeros(head_weight.shape, dtype=np.int64),
            8,
            -128,
            127,
        )
        self._head_bias = MasterTensor(
            head_bias << head_bias_fractional_bits,
            np.zeros(head_bias.shape, dtype=np.int64),
            head_bias_fractional_bits,
            -(1 << 15),
            (1 << 15) - 1,
        )
        self._deployed_head_weight = head_weight.astype(np.int8)
        self._deployed_head_bias = head_bias.astype(np.int64)
        self.optimizer_step = 0

    @property
    def projection_names(self) -> tuple[str, ...]:
        return tuple(self._projection_static)

    def projection(self, name: str) -> ProjectionParameters:
        static = self._projection_static[name]
        return ProjectionParameters(
            static.name,
            self._deployed_projection_weight[name],
            self._deployed_projection_bias[name],
            static.multiplier_q15,
            static.binary_point_shift,
            static.compute_opcode,
            static.source_weight_shape,
            static.output_fractional_bits,
        )

    @property
    def head(self) -> HeadParameters:
        return HeadParameters(self._deployed_head_weight, self._deployed_head_bias)

    @staticmethod
    def _optimizer_coefficients(
        tensor: MasterTensor,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> tuple[int, int, int, int]:
        if not 0.0 <= learning_rate:
            raise ValueError("learning_rate must be nonnegative")
        if not 0.0 <= momentum <= 1.0 or not 0.0 <= weight_decay:
            raise ValueError("momentum must be in [0,1] and weight_decay nonnegative")
        learning_rate_q16 = int(round(learning_rate * (1 << 16)))
        momentum_q16 = int(round(momentum * (1 << 16)))
        decay_q16 = int(round(weight_decay * (1 << (31 - tensor.fractional_bits))))
        update_right_shift = 31 - tensor.fractional_bits
        if learning_rate > 0.0 and learning_rate_q16 == 0:
            raise ValueError("learning_rate is below the Q16 optimizer resolution")
        if learning_rate_q16 > 131071:
            raise OverflowError("Q16 learning rate exceeds the RTL field")
        if momentum_q16 > 65536 or decay_q16 > 65536:
            raise OverflowError("momentum or master-format weight decay exceeds the RTL field")
        return learning_rate_q16, momentum_q16, decay_q16, update_right_shift

    def _update_tensor(
        self,
        driver: FullTrainingDriver,
        tensor: MasterTensor,
        gradient: BlockGradient,
        *,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        gradient_clip_q15: int,
        timeout_seconds: float,
    ) -> np.ndarray:
        if gradient.values.shape != tensor.master.shape:
            raise ValueError("parameter gradient shape does not match its master tensor")
        gradient_q15 = gradient_to_optimizer_q15(
            driver,
            gradient,
            clip_q15=gradient_clip_q15,
            timeout_seconds=timeout_seconds,
        )
        coefficients = self._optimizer_coefficients(
            tensor, learning_rate, momentum, weight_decay)
        next_master, next_velocity = execute_sgdw_update(
            driver,
            tensor.master,
            gradient_q15,
            tensor.velocity_q15,
            learning_rate_q16=coefficients[0],
            momentum_q16=coefficients[1],
            weight_decay_q16=coefficients[2],
            update_right_shift=coefficients[3],
            timeout_seconds=timeout_seconds,
        )
        tensor.master = next_master.astype(np.int16)
        tensor.velocity_q15 = next_velocity.astype(np.int32)
        return execute_deploy_master(driver, tensor, timeout_seconds=timeout_seconds)

    def apply_gradients(
        self,
        driver: FullTrainingDriver,
        gradients: Mapping[str, ParameterGradient],
        *,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        gradient_clip_q15: int = 4 << 15,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Update every projection and classifier coefficient through RTL SGDW."""

        expected = set(self.projection_names) | {"head"}
        if set(gradients) != expected:
            missing = sorted(expected - set(gradients))
            extra = sorted(set(gradients) - expected)
            raise ValueError(f"parameter-gradient coverage mismatch: missing={missing}, extra={extra}")
        for name in self.projection_names:
            gradient = gradients[name]
            self._deployed_projection_weight[name] = self._update_tensor(
                driver,
                self._projection_weight[name],
                gradient.weight,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                gradient_clip_q15=gradient_clip_q15,
                timeout_seconds=timeout_seconds,
            ).astype(np.int8)
            self._deployed_projection_bias[name] = self._update_tensor(
                driver,
                self._projection_bias[name],
                gradient.bias,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=0.0,
                gradient_clip_q15=gradient_clip_q15,
                timeout_seconds=timeout_seconds,
            )
        head_gradient = gradients["head"]
        self._deployed_head_weight = self._update_tensor(
            driver,
            self._head_weight,
            head_gradient.weight,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            gradient_clip_q15=gradient_clip_q15,
            timeout_seconds=timeout_seconds,
        ).astype(np.int8)
        self._deployed_head_bias = self._update_tensor(
            driver,
            self._head_bias,
            head_gradient.bias,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=0.0,
            gradient_clip_q15=gradient_clip_q15,
            timeout_seconds=timeout_seconds,
        )
        self.optimizer_step += 1

    def _checkpoint_tensors(self) -> dict[str, np.ndarray]:
        tensors: dict[str, np.ndarray] = {}
        for index, name in enumerate(self.projection_names):
            for kind, tensor in (
                ("weight", self._projection_weight[name]),
                ("bias", self._projection_bias[name]),
            ):
                prefix = f"projection_{index}_{kind}"
                tensors[f"{prefix}_master"] = tensor.master
                tensors[f"{prefix}_velocity"] = tensor.velocity_q15
        tensors["head_weight_master"] = self._head_weight.master
        tensors["head_weight_velocity"] = self._head_weight.velocity_q15
        tensors["head_bias_master"] = self._head_bias.master
        tensors["head_bias_velocity"] = self._head_bias.velocity_q15
        return tensors

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        batch_index: int,
        best_accuracy_percent: float,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Atomically save complete masters, momentum, and resume coordinates."""

        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": CHECKPOINT_FORMAT,
            "state_format": MASTER_FORMAT,
            "base_binary_sha256": self.base_binary_sha256,
            "projection_names": list(self.projection_names),
            "projection_weight_fractional_bits": [
                self._projection_weight[name].fractional_bits for name in self.projection_names
            ],
            "projection_bias_fractional_bits": [
                self._projection_bias[name].fractional_bits for name in self.projection_names
            ],
            "head_weight_fractional_bits": self._head_weight.fractional_bits,
            "head_bias_fractional_bits": self._head_bias.fractional_bits,
            "optimizer_step": self.optimizer_step,
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "best_accuracy_percent": float(best_accuracy_percent),
            "extra": dict(extra or {}),
        }
        temporary = target.with_name(target.name + ".tmp")
        payload = self._checkpoint_tensors()
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def load_checkpoint(self, path: str | Path) -> Mapping[str, Any]:
        """Restore one complete checkpoint only after all metadata validates."""

        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("unsupported training checkpoint format")
            if metadata.get("base_binary_sha256") != self.base_binary_sha256:
                raise ValueError("checkpoint was initialized from a different parameter binary")
            if tuple(metadata.get("projection_names", ())) != self.projection_names:
                raise ValueError("checkpoint projection order does not match this graph")
            expected_weight_fractional = [
                self._projection_weight[name].fractional_bits for name in self.projection_names
            ]
            expected_bias_fractional = [
                self._projection_bias[name].fractional_bits for name in self.projection_names
            ]
            if metadata.get("projection_weight_fractional_bits") != expected_weight_fractional:
                raise ValueError("checkpoint projection weight Q formats do not match")
            if metadata.get("projection_bias_fractional_bits") != expected_bias_fractional:
                raise ValueError("checkpoint projection bias Q formats do not match")
            if int(metadata.get("head_weight_fractional_bits", -1)) != self._head_weight.fractional_bits:
                raise ValueError("checkpoint head weight Q format does not match")
            if int(metadata.get("head_bias_fractional_bits", -1)) != self._head_bias.fractional_bits:
                raise ValueError("checkpoint head bias Q format does not match")
            staged: list[tuple[MasterTensor, np.ndarray, np.ndarray]] = []
            for index, name in enumerate(self.projection_names):
                for kind, tensor in (
                    ("weight", self._projection_weight[name]),
                    ("bias", self._projection_bias[name]),
                ):
                    prefix = f"projection_{index}_{kind}"
                    master = np.asarray(archive[f"{prefix}_master"], dtype=np.int64)
                    velocity = np.asarray(archive[f"{prefix}_velocity"], dtype=np.int64)
                    if master.shape != tensor.master.shape or velocity.shape != tensor.master.shape:
                        raise ValueError(f"checkpoint tensor shape mismatch for {name} {kind}")
                    if (np.any(master < -(1 << 15)) or np.any(master > (1 << 15) - 1)
                            or np.any(velocity < -(1 << 23))
                            or np.any(velocity > (1 << 23) - 1)):
                        raise ValueError(f"checkpoint tensor range mismatch for {name} {kind}")
                    staged.append((tensor, master.copy(), velocity.copy()))
            for prefix, tensor in (
                ("head_weight", self._head_weight),
                ("head_bias", self._head_bias),
            ):
                master = np.asarray(archive[f"{prefix}_master"], dtype=np.int64)
                velocity = np.asarray(archive[f"{prefix}_velocity"], dtype=np.int64)
                if master.shape != tensor.master.shape or velocity.shape != tensor.master.shape:
                    raise ValueError(f"checkpoint tensor shape mismatch for {prefix}")
                if (np.any(master < -(1 << 15)) or np.any(master > (1 << 15) - 1)
                        or np.any(velocity < -(1 << 23))
                        or np.any(velocity > (1 << 23) - 1)):
                    raise ValueError(f"checkpoint tensor range mismatch for {prefix}")
                staged.append((tensor, master.copy(), velocity.copy()))
        for tensor, master, velocity in staged:
            tensor.master = master.astype(np.int16)
            tensor.velocity_q15 = velocity.astype(np.int32)
        self.optimizer_step = int(metadata["optimizer_step"])
        for name in self.projection_names:
            weight = self._projection_weight[name]
            bias = self._projection_bias[name]
            self._deployed_projection_weight[name] = np.clip(
                round_shift_signed(weight.master, weight.fractional_bits), -128, 127
            ).astype(np.int8)
            self._deployed_projection_bias[name] = np.clip(
                round_shift_signed(bias.master, bias.fractional_bits),
                -(1 << 15),
                (1 << 15) - 1,
            ).astype(np.int64)
        self._deployed_head_weight = np.clip(
            round_shift_signed(self._head_weight.master, self._head_weight.fractional_bits),
            -128,
            127,
        ).astype(np.int8)
        self._deployed_head_bias = np.clip(
            round_shift_signed(self._head_bias.master, self._head_bias.fractional_bits),
            -(1 << 15),
            (1 << 15) - 1,
        ).astype(np.int64)
        return metadata


@dataclass(frozen=True)
class LifTraceCache:
    """Forward values retained for one exact hard-reset LIF reverse pass."""

    input_q10: np.ndarray
    spikes: np.ndarray
    pre_reset_membrane: np.ndarray
    threshold_q10: int


@dataclass(frozen=True)
class TokenizerLayerCache:
    """One tokenizer projection and the LIF/pool operators that precede it."""

    index: int
    projection_input: np.ndarray
    projection_output: np.ndarray
    lif: LifTraceCache | None
    pooled: bool


@dataclass(frozen=True)
class AttentionCache:
    """All state needed to reverse one spike-driven self-attention branch."""

    block_index: int
    input_shape: tuple[int, ...]
    projection_lif: LifTraceCache
    token_spikes: np.ndarray
    query_lif: LifTraceCache
    key_lif: LifTraceCache
    value_lif: LifTraceCache
    query_heads: np.ndarray
    key_heads: np.ndarray
    value_heads: np.ndarray
    scores: np.ndarray
    attention_values: np.ndarray
    attention_lif: LifTraceCache
    projection_input: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class MlpCache:
    """Forward state for the two learned MLP projections and their LIF nodes."""

    block_index: int
    input_shape: tuple[int, ...]
    input_lif: LifTraceCache
    first_projection_input: np.ndarray
    hidden_lif: LifTraceCache
    second_projection_input: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class TransformerBlockCache:
    """Attention, MLP, and the two residual values of one transformer block."""

    input_value: np.ndarray
    attention: AttentionCache
    after_attention: np.ndarray
    mlp: MlpCache
    output: np.ndarray


@dataclass(frozen=True)
class SpikingformerTrainingCache:
    """Complete forward tape for one batch of the exported integer graph."""

    normalized_image: np.ndarray
    image_u8: np.ndarray
    image_centered: np.ndarray
    tokenizer: tuple[TokenizerLayerCache, ...]
    blocks: tuple[TransformerBlockCache, ...]
    final_value: np.ndarray
    head_preclip: np.ndarray
    head_input: np.ndarray
    logits_accumulator: np.ndarray
    logits: np.ndarray


@dataclass(frozen=True)
class TrainingBackwardResult:
    """Loss, classifier probabilities, all parameter gradients, and input dX."""

    loss: float
    probabilities_q15: np.ndarray
    predicted_class: np.ndarray
    parameter_gradients: Mapping[str, ParameterGradient]
    input_gradient: BlockGradient


@dataclass(frozen=True)
class TrainingStepResult:
    """Observable result of one complete forward/backward/SGDW transaction."""

    loss: float
    probabilities_q15: np.ndarray
    predicted_class: np.ndarray
    optimizer_step: int


class SpikingformerOnDeviceTrainer:
    """Execute and train the complete exported Spikingformer graph on NeuraEdge."""

    def __init__(
        self,
        state: TrainableSpikingformerState,
        driver: FullTrainingDriver,
        *,
        transformer_blocks: int = 4,
        attention_heads: int = 12,
        timesteps: int = 4,
        input_shape: tuple[int, int, int] = (3, 32, 32),
        layer_timeout_seconds: float = 120.0,
    ) -> None:
        if transformer_blocks <= 0 or attention_heads <= 0 or timesteps <= 0:
            raise ValueError("block, head, and timestep counts must be positive")
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise ValueError("input_shape must contain positive C,H,W dimensions")
        if layer_timeout_seconds <= 0:
            raise ValueError("layer_timeout_seconds must be positive")
        self.state = state
        self.driver = driver
        self.projections = SpikingformerProjectionExecutor(state, driver)
        self.transformer_blocks = int(transformer_blocks)
        self.attention_heads = int(attention_heads)
        self.timesteps = int(timesteps)
        self.input_shape = tuple(map(int, input_shape))
        self.layer_timeout_seconds = float(layer_timeout_seconds)
        self._surrogate_lut_loaded = False
        fixed = state.manifest.get("fixed_point")
        if not isinstance(fixed, Mapping):
            raise ValueError("trainable state has no fixed-point manifest")
        self.fractional_bits = int(fixed["activation_fractional_bits"])
        self.fixed_scale = 1 << self.fractional_bits
        if self.fixed_scale % 8:
            raise ValueError("attention scaling requires at least three activation fractional bits")
        attention_scale = self.fixed_scale // 8
        if attention_scale <= 0 or attention_scale & (attention_scale - 1):
            raise ValueError("attention scaling must be an exact power of two")
        self.attention_scale_shift = attention_scale.bit_length() - 1
        self.image_scale = float(fixed["image_input_scale"])
        self.image_zero_point = int(fixed["image_input_zero_point"])
        self.head_activation_scale = float(fixed["head_activation_scale"])
        required = {
            *(f"patch_embed.block{index}" for index in range(5)),
            *(f"block.{block}.attn.{name}"
              for block in range(self.transformer_blocks)
              for name in ("q", "k", "v", "proj")),
            *(f"block.{block}.mlp.fc{layer}"
              for block in range(self.transformer_blocks) for layer in (1, 2)),
        }
        missing = required - set(state.projection_names)
        if missing:
            raise ValueError(f"trainable state is missing projections: {sorted(missing)}")
        accumulator_scale = float(state.manifest["head"]["accumulator_scale"])
        ideal_difference = math.log(2.0) / accumulator_scale
        self.softmax_difference_shift = int(round(math.log2(ideal_difference)))
        if not 0 <= self.softmax_difference_shift <= 31:
            raise ValueError("head accumulator scale is outside the hardware softmax range")

    def _lif_trace(self, values: np.ndarray, *, threshold: float = 1.0) -> LifTraceCache:
        threshold_q10 = int(round(threshold * self.fixed_scale))
        spikes, membrane = lif_sequence_with_trace_on_accelerator(
            self.driver,
            values,
            threshold_q10=threshold_q10,
            leak_shift=1,
            reset_q10=0,
            timeout_seconds=self.layer_timeout_seconds,
        )
        return LifTraceCache(
            np.asarray(values, dtype=np.int64),
            spikes,
            membrane,
            threshold_q10,
        )

    def _dynamic_binary_gemm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return compile_spike_gemm(left, right).execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)

    def _dynamic_u8_gemm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return compile_u8_gemm(left, right).execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)

    def _global_sum(self, value: np.ndarray) -> np.ndarray:
        """Reduce time and space through a pairwise fabric ADD tree."""

        timesteps, batch, channels, height, width = value.shape
        reduction = value.transpose(0, 3, 4, 1, 2).reshape(
            timesteps * height * width, batch, channels)
        conservative_bound = int(np.abs(reduction).max(initial=0)) * reduction.shape[0]
        if conservative_bound > (1 << 31) - 1:
            raise OverflowError("global pooling could saturate the signed-32 vector tree")
        candidates = [reduction[index] for index in range(reduction.shape[0])]
        while len(candidates) > 1:
            reduced: list[np.ndarray] = []
            for index in range(0, len(candidates) - 1, 2):
                reduced.append(execute_vector_tensor(
                    self.driver,
                    VectorOperation.ADD_SAT,
                    candidates[index],
                    candidates[index + 1],
                    timeout_seconds=self.layer_timeout_seconds,
                ))
            if len(candidates) & 1:
                reduced.append(candidates[-1])
            candidates = reduced
        return candidates[0]

    def _tokenizer_forward(
        self, image_u8: np.ndarray, image_centered: np.ndarray
    ) -> tuple[np.ndarray, tuple[TokenizerLayerCache, ...]]:
        batch = image_u8.shape[0]
        first = self.projections.conv2d(
            "patch_embed.block0",
            image_u8,
            activation_zero_point=self.image_zero_point,
            padding=1,
            timeout_seconds=self.layer_timeout_seconds,
        )
        caches: list[TokenizerLayerCache] = [
            TokenizerLayerCache(0, image_centered, first, None, False)
        ]
        value = np.repeat(first[np.newaxis], self.timesteps, axis=0)
        for index in (1, 2, 3, 4):
            lif = self._lif_trace(value)
            flat_spikes = lif.spikes.reshape(-1, *lif.spikes.shape[2:])
            if index >= 3:
                projection_input = binary_max_pool2d_on_accelerator(
                    self.driver,
                    flat_spikes,
                    timeout_seconds=self.layer_timeout_seconds,
                )
            else:
                projection_input = flat_spikes
            projected = self.projections.conv2d(
                f"patch_embed.block{index}",
                projection_input,
                padding=1,
                timeout_seconds=self.layer_timeout_seconds,
            )
            caches.append(TokenizerLayerCache(
                index,
                projection_input,
                projected,
                lif,
                index >= 3,
            ))
            value = projected.reshape(
                self.timesteps, batch, *projected.shape[1:])
        return value, tuple(caches)

    def _attention_forward(self, block_index: int, value: np.ndarray) -> AttentionCache:
        timesteps, batch, channels, height, width = value.shape
        tokens = height * width
        projection_lif = self._lif_trace(value)
        token_spikes = projection_lif.spikes.reshape(
            timesteps, batch, channels, tokens).transpose(0, 1, 3, 2)
        rows = token_spikes.reshape(timesteps * batch * tokens, channels)
        prefix = f"block.{block_index}.attn"

        def projected_lif(name: str) -> LifTraceCache:
            projected = self.projections.matrix(
                f"{prefix}.{name}", rows,
                timeout_seconds=self.layer_timeout_seconds)
            shaped = projected.reshape(timesteps, batch, tokens, channels)
            return self._lif_trace(shaped)

        query_lif = projected_lif("q")
        key_lif = projected_lif("k")
        value_lif = projected_lif("v")
        if channels % self.attention_heads:
            raise ValueError("attention channels must divide evenly into the head count")
        head_dim = channels // self.attention_heads

        def heads(spikes: np.ndarray) -> np.ndarray:
            return spikes.reshape(
                timesteps, batch, tokens, self.attention_heads, head_dim
            ).transpose(0, 1, 3, 2, 4)

        query_heads = heads(query_lif.spikes)
        key_heads = heads(key_lif.spikes)
        value_heads = heads(value_lif.spikes)
        scores = np.empty(
            (timesteps, batch, self.attention_heads, tokens, tokens),
            dtype=np.int64,
        )
        attention_values = np.empty(
            (timesteps, batch, self.attention_heads, tokens, head_dim),
            dtype=np.int64,
        )
        for timestep in range(timesteps):
            for batch_index in range(batch):
                for head in range(self.attention_heads):
                    score = self._dynamic_binary_gemm(
                        query_heads[timestep, batch_index, head],
                        key_heads[timestep, batch_index, head].T,
                    )
                    if score.min() < 0 or score.max() > 255:
                        raise OverflowError("binary QK score exceeds the unsigned AV contract")
                    scores[timestep, batch_index, head] = score
                    attention_values[timestep, batch_index, head] = self._dynamic_u8_gemm(
                        score.astype(np.uint8),
                        value_heads[timestep, batch_index, head].astype(np.uint8),
                    )
        attention_q10 = attention_values * (1 << self.attention_scale_shift)
        merged = attention_q10.transpose(0, 1, 2, 4, 3).reshape(
            timesteps, batch, channels, tokens)
        attention_lif = self._lif_trace(merged, threshold=0.5)
        projection_input = attention_lif.spikes.transpose(0, 1, 3, 2).reshape(
            timesteps * batch * tokens, channels)
        output_rows = self.projections.matrix(
            f"{prefix}.proj",
            projection_input,
            timeout_seconds=self.layer_timeout_seconds,
        )
        output = output_rows.reshape(
            timesteps, batch, tokens, channels).transpose(0, 1, 3, 2).reshape(
                timesteps, batch, channels, height, width)
        return AttentionCache(
            block_index,
            tuple(value.shape),
            projection_lif,
            token_spikes,
            query_lif,
            key_lif,
            value_lif,
            query_heads,
            key_heads,
            value_heads,
            scores,
            attention_values,
            attention_lif,
            projection_input,
            output,
        )

    def _mlp_forward(self, block_index: int, value: np.ndarray) -> MlpCache:
        timesteps, batch, channels, height, width = value.shape
        tokens = height * width
        input_lif = self._lif_trace(value)
        first_input = input_lif.spikes.reshape(
            timesteps, batch, channels, tokens).transpose(0, 1, 3, 2).reshape(
                timesteps * batch * tokens, channels)
        hidden_rows = self.projections.matrix(
            f"block.{block_index}.mlp.fc1",
            first_input,
            timeout_seconds=self.layer_timeout_seconds,
        )
        hidden_channels = hidden_rows.shape[1]
        hidden = hidden_rows.reshape(
            timesteps, batch, tokens, hidden_channels).transpose(0, 1, 3, 2)
        hidden_lif = self._lif_trace(hidden)
        second_input = hidden_lif.spikes.transpose(0, 1, 3, 2).reshape(
            timesteps * batch * tokens, hidden_channels)
        output_rows = self.projections.matrix(
            f"block.{block_index}.mlp.fc2",
            second_input,
            timeout_seconds=self.layer_timeout_seconds,
        )
        output = output_rows.reshape(
            timesteps, batch, tokens, channels).transpose(0, 1, 3, 2).reshape(
                timesteps, batch, channels, height, width)
        return MlpCache(
            block_index,
            tuple(value.shape),
            input_lif,
            first_input,
            hidden_lif,
            second_input,
            output,
        )

    def forward(self, normalized_image) -> SpikingformerTrainingCache:
        """Run the full graph and retain every state required by reverse mode."""

        image = np.asarray(normalized_image, dtype=np.float32)
        if image.ndim != 4 or tuple(image.shape[1:]) != self.input_shape or image.shape[0] == 0:
            raise ValueError(f"input must have shape [batch,{','.join(map(str, self.input_shape))}]")
        image_u8 = np.rint(image / self.image_scale + self.image_zero_point)
        image_u8 = np.clip(image_u8, 0, 255).astype(np.uint8)
        image_centered = image_u8.astype(np.int64) - self.image_zero_point
        self.driver.execute_program(
            (ProgramRecord(clear_status_command()),),
            timeout_seconds=self.layer_timeout_seconds,
        )
        value, tokenizer = self._tokenizer_forward(image_u8, image_centered)
        block_caches: list[TransformerBlockCache] = []
        for block_index in range(self.transformer_blocks):
            block_input = value
            attention = self._attention_forward(block_index, block_input)
            after_attention = execute_vector_tensor(
                self.driver,
                VectorOperation.ADD_SAT,
                block_input,
                attention.output,
                timeout_seconds=self.layer_timeout_seconds,
            )
            mlp = self._mlp_forward(block_index, after_attention)
            value = execute_vector_tensor(
                self.driver,
                VectorOperation.ADD_SAT,
                after_attention,
                mlp.output,
                timeout_seconds=self.layer_timeout_seconds,
            )
            block_caches.append(TransformerBlockCache(
                block_input, attention, after_attention, mlp, value))

        pooled_sum = self._global_sum(value.astype(np.int64))
        divisor = value.shape[0] * value.shape[3] * value.shape[4]
        denominator = divisor * self.fixed_scale * self.head_activation_scale
        rounded_denominator = int(round(denominator))
        if not math.isclose(denominator, rounded_denominator, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("head pooling denominator is not an exact integer")
        if rounded_denominator <= 0 or rounded_denominator & (rounded_denominator - 1):
            raise ValueError("head pooling denominator must be a positive power of two")
        head_preclip = execute_power_of_two_scale(
            self.driver,
            pooled_sum,
            -(rounded_denominator.bit_length() - 1),
            timeout_seconds=self.layer_timeout_seconds,
        )
        head_input = execute_vector_tensor(
            self.driver,
            VectorOperation.CLAMP,
            head_preclip,
            clamp_minimum=-128,
            clamp_maximum=127,
            timeout_seconds=self.layer_timeout_seconds,
        )
        head = self.state.head
        head_program = compile_u8_gemm(
            (head_input + 128).astype(np.uint8),
            (head.weight_k_by_n.astype(np.int16) + 128).astype(np.uint8),
            activation_zero_point=128,
            weight_zero_point=128,
            bias=head.bias_accumulator_units,
        )
        logits_accumulator = head_program.execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)
        accumulator_scale = float(self.state.manifest["head"]["accumulator_scale"])
        logits = logits_accumulator.astype(np.float64) * accumulator_scale
        return SpikingformerTrainingCache(
            image,
            image_u8,
            image_centered,
            tokenizer,
            tuple(block_caches),
            value,
            head_preclip,
            head_input,
            logits_accumulator,
            logits,
        )

    def _lif_backward(
        self, cache: LifTraceCache, output_gradient: BlockGradient
    ) -> BlockGradient:
        if output_gradient.values.shape != cache.spikes.shape:
            raise ValueError("LIF output gradient does not match its forward trace")
        timesteps = cache.spikes.shape[0]
        membrane = cache.pre_reset_membrane.reshape(timesteps, -1)
        spikes = cache.spikes.reshape(timesteps, -1)
        gradient = output_gradient.values.reshape(timesteps, -1)
        input_gradient, _ = execute_lif_bptt(
            self.driver,
            membrane,
            spikes,
            gradient,
            threshold=cache.threshold_q10,
            reciprocal_tau_q15=1 << 14,
            surrogate_address_shift=max(self.fractional_bits - 5, 0),
            load_surrogate_table=not self._surrogate_lut_loaded,
            timeout_seconds=self.layer_timeout_seconds,
        )
        self._surrogate_lut_loaded = True
        return BlockGradient(
            input_gradient.reshape(cache.input_q10.shape), output_gradient.exponent)

    def _bias_gradient(self, output_gradient: BlockGradient) -> BlockGradient:
        if output_gradient.values.ndim != 2:
            raise ValueError("bias reduction expects a [rows,channels] gradient")
        rows, channels = output_gradient.values.shape
        ones = np.ones((rows, 1), dtype=np.int64)
        zeros = np.zeros((1, channels), dtype=np.int64)
        _, gradient = execute_linear_backward_block(
            self.driver,
            ones,
            zeros,
            output_gradient,
            timeout_seconds=self.layer_timeout_seconds,
        )
        return BlockGradient(gradient.values.reshape(channels), gradient.exponent)

    def _projection_matrix_backward(
        self,
        name: str,
        activations: np.ndarray,
        output_gradient: BlockGradient,
    ) -> tuple[BlockGradient, ParameterGradient]:
        parameters = self.state.projection(name)
        scaled_gradient = backward_requantize_projection(
            self.driver,
            output_gradient,
            parameters,
            timeout_seconds=self.layer_timeout_seconds,
        )
        input_gradient, weight_gradient = execute_linear_backward_block(
            self.driver,
            activations,
            parameters.weight_k_by_n,
            scaled_gradient,
            timeout_seconds=self.layer_timeout_seconds,
        )
        bias_gradient = self._bias_gradient(output_gradient)
        return input_gradient, ParameterGradient(weight_gradient, bias_gradient)

    def _projection_conv2d_backward(
        self,
        name: str,
        activations: np.ndarray,
        output_gradient: BlockGradient,
        *,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
    ) -> tuple[BlockGradient, ParameterGradient]:
        parameters = self.state.projection(name)
        values = output_gradient.values
        if values.ndim != 4:
            raise ValueError("projection convolution gradient must be NCHW")
        batch, channels, height, width = values.shape
        matrix_gradient = output_gradient.transpose(0, 2, 3, 1).reshape(
            batch * height * width, channels)
        scaled_matrix = backward_requantize_projection(
            self.driver,
            matrix_gradient,
            parameters,
            timeout_seconds=self.layer_timeout_seconds,
        )
        scaled_nchw = scaled_matrix.reshape(
            batch, height, width, channels).transpose(0, 3, 1, 2)
        weight_oihw = parameters.weight_k_by_n.T.reshape(parameters.source_weight_shape)
        input_gradient, accumulator_parameter = execute_conv2d_backward_block(
            self.driver,
            activations,
            weight_oihw,
            parameters.bias_q10,
            scaled_nchw,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            timeout_seconds=self.layer_timeout_seconds,
        )
        weight_gradient = accumulator_parameter.weight.values.reshape(
            parameters.source_weight_shape[0], -1).T
        bias_gradient = self._bias_gradient(matrix_gradient)
        return input_gradient, ParameterGradient(
            BlockGradient(weight_gradient, accumulator_parameter.weight.exponent),
            bias_gradient,
        )

    def _token_rows_gradient(
        self, gradient: BlockGradient, channels: int
    ) -> BlockGradient:
        timesteps, batch, _, height, width = gradient.values.shape
        tokens = height * width
        values = gradient.values.reshape(
            timesteps, batch, channels, tokens).transpose(0, 1, 3, 2).reshape(
                timesteps * batch * tokens, channels)
        return BlockGradient(values, gradient.exponent)

    @staticmethod
    def _rows_to_channel_first(
        gradient: BlockGradient,
        shape: tuple[int, ...],
    ) -> BlockGradient:
        timesteps, batch, channels, height, width = shape
        tokens = height * width
        values = gradient.values.reshape(
            timesteps, batch, tokens, channels).transpose(0, 1, 3, 2).reshape(shape)
        return BlockGradient(values, gradient.exponent)

    def _stack_disjoint_gradients(
        self,
        gradients: Sequence[BlockGradient],
        shape: tuple[int, ...],
    ) -> BlockGradient:
        if not gradients:
            raise ValueError("at least one gradient is required for stacking")
        exponent = max(gradient.exponent for gradient in gradients)
        aligned: list[np.ndarray] = []
        for gradient in gradients:
            values = gradient.values
            if gradient.exponent != exponent:
                values = execute_power_of_two_scale(
                    self.driver,
                    values,
                    gradient.exponent - exponent,
                    timeout_seconds=self.layer_timeout_seconds,
                )
            aligned.append(values)
        return BlockGradient(np.stack(aligned).reshape(shape), exponent)

    def _gate_saturating_add_gradient(
        self,
        output_gradient: BlockGradient,
        left: np.ndarray,
        right: np.ndarray,
    ) -> BlockGradient:
        total = np.asarray(left, dtype=np.int64) + np.asarray(right, dtype=np.int64)
        mask = ((total >= -(1 << 31)) & (total <= (1 << 31) - 1)).astype(np.int64)
        gated = execute_vector_tensor(
            self.driver,
            VectorOperation.SELECT_NONZERO,
            mask,
            output_gradient.values,
            timeout_seconds=self.layer_timeout_seconds,
        )
        return BlockGradient(gated, output_gradient.exponent)

    def _mlp_backward(
        self,
        cache: MlpCache,
        output_gradient: BlockGradient,
    ) -> tuple[BlockGradient, dict[str, ParameterGradient]]:
        timesteps, batch, channels, height, width = cache.input_shape
        tokens = height * width
        output_rows = self._token_rows_gradient(output_gradient, channels)
        second_name = f"block.{cache.block_index}.mlp.fc2"
        hidden_spike_rows, second_parameter = self._projection_matrix_backward(
            second_name, cache.second_projection_input, output_rows)
        hidden_channels = cache.hidden_lif.spikes.shape[2]
        hidden_spike_gradient = hidden_spike_rows.reshape(
            timesteps, batch, tokens, hidden_channels).transpose(0, 1, 3, 2)
        hidden_gradient = self._lif_backward(
            cache.hidden_lif,
            hidden_spike_gradient,
        )
        hidden_rows = BlockGradient(
            hidden_gradient.values.transpose(0, 1, 3, 2).reshape(
                timesteps * batch * tokens, hidden_channels),
            hidden_gradient.exponent,
        )
        first_name = f"block.{cache.block_index}.mlp.fc1"
        input_spike_rows, first_parameter = self._projection_matrix_backward(
            first_name, cache.first_projection_input, hidden_rows)
        input_spike_gradient = self._rows_to_channel_first(
            input_spike_rows, cache.input_shape)
        input_gradient = self._lif_backward(cache.input_lif, input_spike_gradient)
        return input_gradient, {
            first_name: first_parameter,
            second_name: second_parameter,
        }

    def _attention_backward(
        self,
        cache: AttentionCache,
        output_gradient: BlockGradient,
    ) -> tuple[BlockGradient, dict[str, ParameterGradient]]:
        timesteps, batch, channels, height, width = cache.input_shape
        tokens = height * width
        head_dim = channels // self.attention_heads
        prefix = f"block.{cache.block_index}.attn"
        output_rows = self._token_rows_gradient(output_gradient, channels)
        attention_spike_rows, projection_parameter = self._projection_matrix_backward(
            f"{prefix}.proj", cache.projection_input, output_rows)
        attention_spike_gradient = BlockGradient(
            attention_spike_rows.values.reshape(
                timesteps, batch, tokens, channels).transpose(0, 1, 3, 2),
            attention_spike_rows.exponent,
        )
        merged_gradient = self._lif_backward(
            cache.attention_lif, attention_spike_gradient)
        attention_value_gradient = BlockGradient(
            merged_gradient.values.reshape(
                timesteps, batch, self.attention_heads, head_dim, tokens
            ).transpose(0, 1, 2, 4, 3),
            merged_gradient.exponent + self.attention_scale_shift,
        )

        score_parts: list[BlockGradient] = []
        value_parts: list[BlockGradient] = []
        for timestep in range(timesteps):
            for batch_index in range(batch):
                for head in range(self.attention_heads):
                    score_gradient, value_gradient = execute_linear_backward_block(
                        self.driver,
                        cache.scores[timestep, batch_index, head],
                        cache.value_heads[timestep, batch_index, head],
                        BlockGradient(
                            attention_value_gradient.values[
                                timestep, batch_index, head],
                            attention_value_gradient.exponent,
                        ),
                        timeout_seconds=self.layer_timeout_seconds,
                    )
                    score_parts.append(score_gradient)
                    value_parts.append(value_gradient)
        score_gradient = self._stack_disjoint_gradients(
            score_parts,
            (timesteps, batch, self.attention_heads, tokens, tokens),
        )
        value_head_gradient = self._stack_disjoint_gradients(
            value_parts,
            (timesteps, batch, self.attention_heads, tokens, head_dim),
        )

        query_parts: list[BlockGradient] = []
        key_parts: list[BlockGradient] = []
        part_index = 0
        for timestep in range(timesteps):
            for batch_index in range(batch):
                for head in range(self.attention_heads):
                    query_gradient, transposed_key_gradient = execute_linear_backward_block(
                        self.driver,
                        cache.query_heads[timestep, batch_index, head],
                        cache.key_heads[timestep, batch_index, head].T,
                        BlockGradient(
                            score_gradient.values[timestep, batch_index, head],
                            score_gradient.exponent,
                        ),
                        timeout_seconds=self.layer_timeout_seconds,
                    )
                    query_parts.append(query_gradient)
                    key_parts.append(BlockGradient(
                        transposed_key_gradient.values.T,
                        transposed_key_gradient.exponent,
                    ))
                    part_index += 1
        query_head_gradient = self._stack_disjoint_gradients(
            query_parts,
            (timesteps, batch, self.attention_heads, tokens, head_dim),
        )
        key_head_gradient = self._stack_disjoint_gradients(
            key_parts,
            (timesteps, batch, self.attention_heads, tokens, head_dim),
        )

        def merge_heads(gradient: BlockGradient) -> BlockGradient:
            return BlockGradient(
                gradient.values.transpose(0, 1, 3, 2, 4).reshape(
                    timesteps, batch, tokens, channels),
                gradient.exponent,
            )

        query_gradient = self._lif_backward(cache.query_lif, merge_heads(query_head_gradient))
        key_gradient = self._lif_backward(cache.key_lif, merge_heads(key_head_gradient))
        value_gradient = self._lif_backward(cache.value_lif, merge_heads(value_head_gradient))
        token_input = cache.token_spikes.reshape(timesteps * batch * tokens, channels)
        branch_input_gradients: list[BlockGradient] = []
        parameters: dict[str, ParameterGradient] = {
            f"{prefix}.proj": projection_parameter,
        }
        for name, gradient in (
            ("q", query_gradient),
            ("k", key_gradient),
            ("v", value_gradient),
        ):
            rows = BlockGradient(
                gradient.values.reshape(timesteps * batch * tokens, channels),
                gradient.exponent,
            )
            input_rows, parameter = self._projection_matrix_backward(
                f"{prefix}.{name}", token_input, rows)
            branch_input_gradients.append(input_rows)
            parameters[f"{prefix}.{name}"] = parameter
        token_gradient = add_block_gradients(
            self.driver,
            branch_input_gradients[0],
            branch_input_gradients[1],
            timeout_seconds=self.layer_timeout_seconds,
        )
        token_gradient = add_block_gradients(
            self.driver,
            token_gradient,
            branch_input_gradients[2],
            timeout_seconds=self.layer_timeout_seconds,
        )
        projection_spike_gradient = self._rows_to_channel_first(
            token_gradient, cache.input_shape)
        input_gradient = self._lif_backward(
            cache.projection_lif, projection_spike_gradient)
        return input_gradient, parameters

    def _block_backward(
        self,
        cache: TransformerBlockCache,
        output_gradient: BlockGradient,
    ) -> tuple[BlockGradient, dict[str, ParameterGradient]]:
        residual_two_gradient = self._gate_saturating_add_gradient(
            output_gradient, cache.after_attention, cache.mlp.output)
        mlp_input_gradient, mlp_parameters = self._mlp_backward(
            cache.mlp, residual_two_gradient)
        after_attention_gradient = add_block_gradients(
            self.driver,
            residual_two_gradient,
            mlp_input_gradient,
            timeout_seconds=self.layer_timeout_seconds,
        )
        residual_one_gradient = self._gate_saturating_add_gradient(
            after_attention_gradient, cache.input_value, cache.attention.output)
        attention_input_gradient, attention_parameters = self._attention_backward(
            cache.attention, residual_one_gradient)
        input_gradient = add_block_gradients(
            self.driver,
            residual_one_gradient,
            attention_input_gradient,
            timeout_seconds=self.layer_timeout_seconds,
        )
        return input_gradient, {**attention_parameters, **mlp_parameters}

    def _sum_timestep_gradients(self, gradient: BlockGradient) -> BlockGradient:
        parts = [
            BlockGradient(gradient.values[timestep], gradient.exponent)
            for timestep in range(gradient.values.shape[0])
        ]
        while len(parts) > 1:
            reduced: list[BlockGradient] = []
            for index in range(0, len(parts) - 1, 2):
                reduced.append(add_block_gradients(
                    self.driver,
                    parts[index],
                    parts[index + 1],
                    timeout_seconds=self.layer_timeout_seconds,
                ))
            if len(parts) & 1:
                reduced.append(parts[-1])
            parts = reduced
        return parts[0]

    def _tokenizer_backward(
        self,
        caches: tuple[TokenizerLayerCache, ...],
        output_gradient: BlockGradient,
    ) -> tuple[BlockGradient, dict[str, ParameterGradient]]:
        gradient = output_gradient
        parameters: dict[str, ParameterGradient] = {}
        for index in range(4, 0, -1):
            cache = caches[index]
            if cache.lif is None:
                raise RuntimeError("nonzero tokenizer layer has no LIF trace")
            flat_gradient = BlockGradient(
                gradient.values.reshape(-1, *gradient.values.shape[2:]),
                gradient.exponent,
            )
            name = f"patch_embed.block{index}"
            projection_input_gradient, parameter = self._projection_conv2d_backward(
                name,
                cache.projection_input,
                flat_gradient,
                padding=1,
            )
            parameters[name] = parameter
            if cache.pooled:
                flat_spikes = cache.lif.spikes.reshape(-1, *cache.lif.spikes.shape[2:])
                spike_gradient, _ = execute_maxpool2d_backward(
                    self.driver,
                    flat_spikes,
                    projection_input_gradient.values,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    timeout_seconds=self.layer_timeout_seconds,
                )
                projection_input_gradient = BlockGradient(
                    spike_gradient, projection_input_gradient.exponent)
            lif_spike_gradient = BlockGradient(
                projection_input_gradient.values.reshape(cache.lif.spikes.shape),
                projection_input_gradient.exponent,
            )
            gradient = self._lif_backward(cache.lif, lif_spike_gradient)

        first_output_gradient = self._sum_timestep_gradients(gradient)
        first_cache = caches[0]
        image_gradient, first_parameter = self._projection_conv2d_backward(
            "patch_embed.block0",
            first_cache.projection_input,
            first_output_gradient,
            padding=1,
        )
        parameters["patch_embed.block0"] = first_parameter
        return image_gradient, parameters

    def backward(
        self,
        cache: SpikingformerTrainingCache,
        labels,
    ) -> TrainingBackwardResult:
        """Reverse every exported operator and produce all trainable gradients."""

        labels_array = np.asarray(labels, dtype=np.int64)
        probabilities = execute_softmax_q15(
            self.driver,
            cache.logits_accumulator,
            difference_shift=self.softmax_difference_shift,
            timeout_seconds=self.layer_timeout_seconds,
        )
        head_output_gradient = execute_cross_entropy_gradient_q15(
            self.driver,
            probabilities,
            labels_array,
            timeout_seconds=self.layer_timeout_seconds,
        )
        head = self.state.head
        head_input_gradient, head_parameter = execute_affine_backward_block(
            self.driver,
            cache.head_input,
            head.weight_k_by_n,
            head.bias_accumulator_units,
            head_output_gradient,
            timeout_seconds=self.layer_timeout_seconds,
        )
        clip_mask = (
            (cache.head_preclip >= -128) & (cache.head_preclip <= 127)
        ).astype(np.int64)
        head_input_values = execute_vector_tensor(
            self.driver,
            VectorOperation.SELECT_NONZERO,
            clip_mask,
            head_input_gradient.values,
            timeout_seconds=self.layer_timeout_seconds,
        )
        head_input_gradient = BlockGradient(
            head_input_values, head_input_gradient.exponent)
        divisor = (
            cache.final_value.shape[0]
            * cache.final_value.shape[3]
            * cache.final_value.shape[4]
        )
        denominator = int(round(divisor * self.fixed_scale * self.head_activation_scale))
        if denominator <= 0 or denominator & (denominator - 1):
            raise ValueError("the exported head pooling denominator must be a power of two")
        pooling_shift = denominator.bit_length() - 1
        final_gradient = BlockGradient(
            np.broadcast_to(
                head_input_gradient.values[np.newaxis, :, :, np.newaxis, np.newaxis],
                cache.final_value.shape,
            ).copy(),
            head_input_gradient.exponent - pooling_shift,
        )

        parameter_gradients: dict[str, ParameterGradient] = {"head": head_parameter}
        gradient = final_gradient
        for block_cache in reversed(cache.blocks):
            gradient, block_parameters = self._block_backward(block_cache, gradient)
            overlap = set(parameter_gradients) & set(block_parameters)
            if overlap:
                raise RuntimeError(f"duplicate block parameter gradients: {sorted(overlap)}")
            parameter_gradients.update(block_parameters)
        image_gradient, tokenizer_parameters = self._tokenizer_backward(
            cache.tokenizer, gradient)
        overlap = set(parameter_gradients) & set(tokenizer_parameters)
        if overlap:
            raise RuntimeError(f"duplicate tokenizer parameter gradients: {sorted(overlap)}")
        parameter_gradients.update(tokenizer_parameters)
        expected = set(self.state.projection_names) | {"head"}
        if set(parameter_gradients) != expected:
            missing = sorted(expected - set(parameter_gradients))
            extra = sorted(set(parameter_gradients) - expected)
            raise RuntimeError(f"reverse graph coverage mismatch: missing={missing}, extra={extra}")

        selected = probabilities[np.arange(probabilities.shape[0]), labels_array]
        loss = float(-np.log(np.maximum(selected, 1) / float(1 << 15)).mean())
        return TrainingBackwardResult(
            loss,
            probabilities,
            np.argmax(cache.logits_accumulator, axis=1),
            parameter_gradients,
            image_gradient,
        )

    def train_step(
        self,
        normalized_image,
        labels,
        *,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        gradient_clip_q15: int = 4 << 15,
    ) -> TrainingStepResult:
        """Execute one full forward, reverse, and all-parameter SGDW update."""

        cache = self.forward(normalized_image)
        backward = self.backward(cache, labels)
        self.state.apply_gradients(
            self.driver,
            backward.parameter_gradients,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            gradient_clip_q15=gradient_clip_q15,
            timeout_seconds=self.layer_timeout_seconds,
        )
        return TrainingStepResult(
            backward.loss,
            backward.probabilities_q15,
            backward.predicted_class,
            self.state.optimizer_step,
        )
