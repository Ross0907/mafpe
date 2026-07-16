"""Executable Spikingformer projection path for the NeuraEdge command ABI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neuraedge_compiler import (
    ProgramExecutor,
    compile_spike_conv2d,
    compile_spike_gemm,
    compile_u8_conv2d,
    compile_u8_gemm,
)
from neuraedge_golden import VectorOperation, round_shift_signed, saturate_signed
from neuraedge_protocol import (
    ProgramRecord,
    clear_status_command,
    pack_vector_beats,
    lif_command,
    pack_lif_beats,
    unpack_lif_result,
    unpack_lif_trace,
    unpack_vector_result,
    vector_command,
)
from spikingformer_parameters import ProjectionParameters, SpikingformerParameterArchive


VECTOR_LANES = 8
LIF_LANES = 32


def vector_binary_on_accelerator(
    driver: ProgramExecutor,
    operation: VectorOperation,
    operand_a: np.ndarray,
    operand_b: np.ndarray,
    *,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Apply a binary vector-ALU operation to two equally shaped tensors."""

    if operation not in {VectorOperation.ADD_SAT, VectorOperation.MAXIMUM}:
        raise ValueError("this helper supports saturating add and maximum")
    a = np.asarray(operand_a, dtype=np.int64)
    b = np.asarray(operand_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError("vector operands must have equal shapes")
    if a.size and (a.min() < -(1 << 31) or a.max() > (1 << 31) - 1
                   or b.min() < -(1 << 31) or b.max() > (1 << 31) - 1):
        raise OverflowError("vector operands do not fit signed 32-bit lanes")
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    records: list[ProgramRecord] = []
    counts: list[int] = []
    command = vector_command(int(operation))
    for start in range(0, len(flat_a), VECTOR_LANES):
        count = min(VECTOR_LANES, len(flat_a) - start)
        lanes_a = flat_a[start:start + count].tolist()
        lanes_b = flat_b[start:start + count].tolist()
        lanes_a.extend([0] * (VECTOR_LANES - count))
        lanes_b.extend([0] * (VECTOR_LANES - count))
        records.append(ProgramRecord(
            command, pack_vector_beats(lanes_a, lanes_b), 4))
        counts.append(count)
    if not records:
        return np.empty(a.shape, dtype=np.int64)
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != 4 * len(records):
        raise RuntimeError(f"vector program returned {len(words)} beats; expected {4 * len(records)}")
    result = np.empty(len(flat_a), dtype=np.int64)
    cursor = 0
    for record_index, count in enumerate(counts):
        lanes = unpack_vector_result(words[4 * record_index:4 * record_index + 4])
        result[cursor:cursor + count] = lanes[:count]
        cursor += count
    return result.reshape(a.shape)


def power_of_two_requantize_on_accelerator(
    driver: ProgramExecutor,
    values: np.ndarray,
    *,
    division_shift: int,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Round ``values / 2**division_shift`` in the flexible vector datapath."""

    source = np.asarray(values, dtype=np.int64)
    if not 0 <= division_shift <= 16:
        raise ValueError("division_shift must be in [0,16]")
    if source.size and (
        source.min() < -(1 << 31) or source.max() > (1 << 31) - 1
    ):
        raise OverflowError("requantizer input must fit signed 32-bit lanes")
    flat = source.reshape(-1)
    records: list[ProgramRecord] = []
    counts: list[int] = []
    command = vector_command(
        int(VectorOperation.FLEX_REQUANTIZE),
        multiplier_q15=1 << 14,
        right_shift=division_shift - 1,
    )
    for start in range(0, flat.size, VECTOR_LANES):
        stop = min(start + VECTOR_LANES, flat.size)
        count = stop - start
        lanes = flat[start:stop].tolist() + [0] * (VECTOR_LANES - count)
        records.append(ProgramRecord(
            command,
            pack_vector_beats(
                lanes,
                [0] * VECTOR_LANES,
                clamp_minimum=clamp_minimum,
                clamp_maximum=clamp_maximum,
            ),
            4,
        ))
        counts.append(count)
    if not records:
        return np.empty(source.shape, dtype=np.int64)
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != 4 * len(records):
        raise RuntimeError("power-of-two requantizer returned an invalid stream length")
    result = np.empty(flat.size, dtype=np.int64)
    cursor = 0
    for record_index, count in enumerate(counts):
        lanes = unpack_vector_result(words[4 * record_index:4 * record_index + 4])
        result[cursor:cursor + count] = lanes[:count]
        cursor += count
    return result.reshape(source.shape)


def global_spatiotemporal_sum_on_accelerator(
    driver: ProgramExecutor,
    values: np.ndarray,
    *,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Reduce a [T,B,C,H,W] tensor to [B,C] through pairwise vector adds."""

    source = np.asarray(values, dtype=np.int64)
    if source.ndim != 5:
        raise ValueError("global sum input must have shape [T,B,C,H,W]")
    timesteps, batch, channels, height, width = source.shape
    reduction = source.transpose(0, 3, 4, 1, 2).reshape(
        timesteps * height * width, batch, channels)
    if int(np.abs(reduction).max(initial=0)) * reduction.shape[0] > (1 << 31) - 1:
        raise OverflowError("global sum can exceed the signed-32 vector accumulator")
    candidates = [reduction[index] for index in range(reduction.shape[0])]
    while len(candidates) > 1:
        reduced: list[np.ndarray] = []
        for index in range(0, len(candidates) - 1, 2):
            reduced.append(vector_binary_on_accelerator(
                driver,
                VectorOperation.ADD_SAT,
                candidates[index],
                candidates[index + 1],
                timeout_seconds=timeout_seconds,
            ))
        if len(candidates) & 1:
            reduced.append(candidates[-1])
        candidates = reduced
    return candidates[0]


def binary_max_pool2d_on_accelerator(
    driver: ProgramExecutor,
    values: np.ndarray,
    *,
    kernel_size: int = 3,
    stride: int = 2,
    padding: int = 1,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Max-pool NCHW binary spikes with pairwise vector-maximum trees."""

    source = np.asarray(values)
    if source.ndim != 4 or np.any((source != 0) & (source != 1)):
        raise ValueError("binary max-pool input must be an NCHW tensor containing only 0/1")
    if kernel_size <= 0 or stride <= 0 or padding < 0:
        raise ValueError("invalid max-pool geometry")
    padded = np.pad(
        source.astype(np.int64),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
    )
    output_height = (source.shape[2] + 2 * padding - kernel_size) // stride + 1
    output_width = (source.shape[3] + 2 * padding - kernel_size) // stride + 1
    candidates = [
        padded[:, :, kernel_y:kernel_y + output_height * stride:stride,
               kernel_x:kernel_x + output_width * stride:stride]
        for kernel_y in range(kernel_size)
        for kernel_x in range(kernel_size)
    ]
    while len(candidates) > 1:
        reduced: list[np.ndarray] = []
        for index in range(0, len(candidates) - 1, 2):
            reduced.append(vector_binary_on_accelerator(
                driver,
                VectorOperation.MAXIMUM,
                candidates[index],
                candidates[index + 1],
                timeout_seconds=timeout_seconds,
            ))
        if len(candidates) & 1:
            reduced.append(candidates[-1])
        candidates = reduced
    return candidates[0].astype(np.uint8)


def lif_sequence_reference(
    values_q10: np.ndarray,
    *,
    threshold_q10: int = 1 << 10,
    leak_shift: int = 1,
    reset_q10: int = 0,
) -> np.ndarray:
    """Bit-exact fixed-point LIF sequence for an arbitrary tensor shape."""

    values = np.asarray(values_q10, dtype=np.int64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("LIF input must have shape [timesteps, ...]")
    if not 0 <= leak_shift <= 31:
        raise ValueError("leak_shift must be in [0,31]")
    membrane = np.full(values.shape[1:], reset_q10, dtype=np.int64)
    spikes = np.zeros(values.shape, dtype=np.uint8)
    for timestep in range(values.shape[0]):
        current = np.clip(values[timestep] >> 1, -(1 << 15), (1 << 15) - 1)
        leaked = membrane - (membrane >> leak_shift)
        membrane = saturate_signed(leaked + current, 24)
        fired = membrane >= threshold_q10
        spikes[timestep] = fired
        membrane = np.where(fired, reset_q10, membrane)
    return spikes


def lif_sequence_with_trace_reference(
    values_q10: np.ndarray,
    *,
    threshold_q10: int = 1 << 10,
    leak_shift: int = 1,
    reset_q10: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return spikes and the signed-24 membrane immediately before hard reset."""

    values = np.asarray(values_q10, dtype=np.int64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("LIF input must have shape [timesteps, ...]")
    if not 0 <= leak_shift <= 31:
        raise ValueError("leak_shift must be in [0,31]")
    membrane = np.full(values.shape[1:], reset_q10, dtype=np.int64)
    spikes = np.zeros(values.shape, dtype=np.uint8)
    membrane_trace = np.zeros(values.shape, dtype=np.int64)
    for timestep in range(values.shape[0]):
        current = np.clip(values[timestep] >> 1, -(1 << 15), (1 << 15) - 1)
        leaked = membrane - (membrane >> leak_shift)
        membrane = saturate_signed(leaked + current, 24)
        membrane_trace[timestep] = membrane
        fired = membrane >= threshold_q10
        spikes[timestep] = fired
        membrane = np.where(fired, reset_q10, membrane)
    return spikes, membrane_trace


def lif_sequence_on_accelerator(
    driver: ProgramExecutor,
    values_q10: np.ndarray,
    *,
    threshold_q10: int = 1 << 10,
    leak_shift: int = 1,
    reset_q10: int = 0,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Schedule every logical LIF state through the physical 32-lane bank."""

    values = np.asarray(values_q10, dtype=np.int64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("LIF input must have shape [timesteps, ...]")
    if values.size and (values.min() < -(1 << 63) or values.max() > (1 << 63) - 1):
        raise OverflowError("LIF input exceeds signed 64-bit preparation arithmetic")
    timesteps = values.shape[0]
    flat = values.reshape(timesteps, -1)
    records: list[ProgramRecord] = []
    bindings: list[tuple[int, int, int]] = []
    for element_start in range(0, flat.shape[1], LIF_LANES):
        element_count = min(LIF_LANES, flat.shape[1] - element_start)
        for timestep in range(timesteps):
            currents = np.clip(
                flat[timestep, element_start:element_start + element_count] >> 1,
                -(1 << 15),
                (1 << 15) - 1,
            ).astype(np.int64).tolist()
            currents.extend([0] * (LIF_LANES - element_count))
            records.append(ProgramRecord(
                lif_command(
                    leak_shift,
                    threshold_q10,
                    reset_q10,
                    clear_state=timestep == 0,
                ),
                pack_lif_beats(currents),
                1,
            ))
            bindings.append((timestep, element_start, element_count))
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != len(records):
        raise RuntimeError(f"LIF program returned {len(words)} beats; expected {len(records)}")
    output = np.zeros(flat.shape, dtype=np.uint8)
    for word, (timestep, element_start, element_count) in zip(words, bindings):
        spike_word, _ = unpack_lif_result(word)
        for lane in range(element_count):
            output[timestep, element_start + lane] = (spike_word >> lane) & 1
    return output.reshape(values.shape)


def lif_sequence_with_trace_on_accelerator(
    driver: ProgramExecutor,
    values_q10: np.ndarray,
    *,
    threshold_q10: int = 1 << 10,
    leak_shift: int = 1,
    reset_q10: int = 0,
    timeout_seconds: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run LIF in fabric and retain the pre-reset membrane required by BPTT."""

    values = np.asarray(values_q10, dtype=np.int64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("LIF input must have shape [timesteps, ...]")
    if values.size and (values.min() < -(1 << 63) or values.max() > (1 << 63) - 1):
        raise OverflowError("LIF input exceeds signed 64-bit preparation arithmetic")
    timesteps = values.shape[0]
    flat = values.reshape(timesteps, -1)
    records: list[ProgramRecord] = []
    bindings: list[tuple[int, int, int]] = []
    for element_start in range(0, flat.shape[1], LIF_LANES):
        element_count = min(LIF_LANES, flat.shape[1] - element_start)
        for timestep in range(timesteps):
            currents = np.clip(
                flat[timestep, element_start:element_start + element_count] >> 1,
                -(1 << 15),
                (1 << 15) - 1,
            ).astype(np.int64).tolist()
            currents.extend([0] * (LIF_LANES - element_count))
            records.append(ProgramRecord(
                lif_command(
                    leak_shift,
                    threshold_q10,
                    reset_q10,
                    clear_state=timestep == 0,
                    trace_state=True,
                ),
                pack_lif_beats(currents),
                13,
            ))
            bindings.append((timestep, element_start, element_count))
    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    expected_words = 13 * len(records)
    if len(words) != expected_words:
        raise RuntimeError(
            f"traced LIF program returned {len(words)} beats; expected {expected_words}")
    spikes = np.zeros(flat.shape, dtype=np.uint8)
    membrane_trace = np.zeros(flat.shape, dtype=np.int64)
    for record_index, (timestep, element_start, element_count) in enumerate(bindings):
        start = record_index * 13
        spike_word, _, membranes = unpack_lif_trace(words[start:start + 13])
        for lane in range(element_count):
            spikes[timestep, element_start + lane] = (spike_word >> lane) & 1
            membrane_trace[timestep, element_start + lane] = membranes[lane]
    return spikes.reshape(values.shape), membrane_trace.reshape(values.shape)


def requantize_projection_reference(
    accumulators: np.ndarray,
    parameters: ProjectionParameters,
) -> np.ndarray:
    """Apply the exported per-channel Q15 fields exactly like vector operation 9."""

    values = np.asarray(accumulators, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != parameters.weight_k_by_n.shape[1]:
        raise ValueError("accumulators must have shape [rows, projection output channels]")
    result = np.empty_like(values)
    for channel in range(values.shape[1]):
        effective_shift = 15 + int(parameters.binary_point_shift[channel])
        product = values[:, channel] * int(parameters.multiplier_q15[channel])
        requantized = round_shift_signed(product, effective_shift)
        requantized += int(parameters.bias_q10[channel])
        result[:, channel] = saturate_signed(requantized, 24)
    return result


def requantize_projection_on_accelerator(
    driver: ProgramExecutor,
    accumulators: np.ndarray,
    parameters: ProjectionParameters,
    *,
    timeout_seconds: float = 30.0,
) -> np.ndarray:
    """Execute per-channel folded BN scaling through the eight-lane vector RTL."""

    values = np.asarray(accumulators, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != parameters.weight_k_by_n.shape[1]:
        raise ValueError("accumulators must have shape [rows, projection output channels]")
    if values.size and (values.min() < -(1 << 31) or values.max() > (1 << 31) - 1):
        raise OverflowError("projection accumulators do not fit the vector ALU's signed 32-bit lanes")

    records: list[ProgramRecord] = []
    bindings: list[tuple[int, int, int]] = []
    zero = [0] * VECTOR_LANES
    rows, channels = values.shape
    for channel in range(channels):
        command = vector_command(
            int(VectorOperation.FLEX_REQUANTIZE),
            multiplier_q15=int(parameters.multiplier_q15[channel]),
            right_shift=int(parameters.binary_point_shift[channel]),
            output_zero_point=int(parameters.bias_q10[channel]),
        )
        for row_start in range(0, rows, VECTOR_LANES):
            row_count = min(VECTOR_LANES, rows - row_start)
            lanes = values[row_start:row_start + row_count, channel].tolist()
            lanes.extend([0] * (VECTOR_LANES - row_count))
            records.append(ProgramRecord(
                command,
                pack_vector_beats(
                    lanes,
                    zero,
                    clamp_minimum=-(1 << 23),
                    clamp_maximum=(1 << 23) - 1,
                ),
                4,
            ))
            bindings.append((row_start, row_count, channel))

    words = driver.execute_program(records, timeout_seconds=timeout_seconds)
    if len(words) != 4 * len(records):
        raise RuntimeError(
            f"requantizer returned {len(words)} beats; expected {4 * len(records)}")
    result = np.empty((rows, channels), dtype=np.int64)
    for record_index, (row_start, row_count, channel) in enumerate(bindings):
        lanes = unpack_vector_result(words[4 * record_index:4 * record_index + 4])
        result[row_start:row_start + row_count, channel] = lanes[:row_count]
    return result


@dataclass
class SpikingformerProjectionExecutor:
    """Run exported Conv-BN projections through tensor and vector hardware."""

    archive: SpikingformerParameterArchive
    driver: ProgramExecutor

    @staticmethod
    def _encoded_dense_weights(weight: np.ndarray) -> np.ndarray:
        return (weight.astype(np.int16) + 128).astype(np.uint8)

    def matrix(
        self,
        name: str,
        activations: np.ndarray,
        *,
        activation_zero_point: int = 0,
        timeout_seconds: float = 30.0,
    ) -> np.ndarray:
        """Execute one exported Conv1d-style projection as a two-dimensional matrix."""

        parameters = self.archive.projection(name)
        values = np.asarray(activations)
        if values.ndim != 2 or values.shape[1] != parameters.weight_k_by_n.shape[0]:
            raise ValueError("matrix activations do not match the projection reduction dimension")
        if parameters.compute_opcode == "COMPUTE_BINARY":
            compiled = compile_spike_gemm(values, parameters.weight_k_by_n)
        elif parameters.compute_opcode == "COMPUTE_DENSE_OUTER":
            compiled = compile_u8_gemm(
                values,
                self._encoded_dense_weights(parameters.weight_k_by_n),
                activation_zero_point=activation_zero_point,
                weight_zero_point=128,
            )
        else:
            raise ValueError(f"unsupported projection opcode {parameters.compute_opcode!r}")
        accumulators = compiled.execute_batched(
            self.driver, timeout_seconds=timeout_seconds)
        return requantize_projection_on_accelerator(
            self.driver, accumulators, parameters, timeout_seconds=timeout_seconds)

    def conv2d(
        self,
        name: str,
        activations: np.ndarray,
        *,
        activation_zero_point: int = 0,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        timeout_seconds: float = 30.0,
    ) -> np.ndarray:
        """Execute one exported Conv2d-BN pair and return signed Q10 NCHW data."""

        parameters = self.archive.projection(name)
        if len(parameters.source_weight_shape) != 4:
            raise ValueError(f"projection {name} is not Conv2d")
        weight = parameters.weight_k_by_n.T.reshape(parameters.source_weight_shape)
        values = np.asarray(activations)
        if parameters.compute_opcode == "COMPUTE_BINARY":
            compiled = compile_spike_conv2d(
                values, weight, stride=stride, padding=padding,
                dilation=dilation, groups=groups)
        elif parameters.compute_opcode == "COMPUTE_DENSE_OUTER":
            compiled = compile_u8_conv2d(
                values,
                self._encoded_dense_weights(weight),
                activation_zero_point=activation_zero_point,
                weight_zero_point=128,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            raise ValueError(f"unsupported projection opcode {parameters.compute_opcode!r}")
        accumulator_nchw = compiled.execute_batched(
            self.driver, timeout_seconds=timeout_seconds)
        batch, channels, height, width = accumulator_nchw.shape
        matrix = accumulator_nchw.transpose(0, 2, 3, 1).reshape(-1, channels)
        quantized = requantize_projection_on_accelerator(
            self.driver, matrix, parameters, timeout_seconds=timeout_seconds)
        return quantized.reshape(batch, height, width, channels).transpose(0, 3, 1, 2)


@dataclass(frozen=True)
class SpikingformerInferenceResult:
    """Classifier result in both exact accumulator and represented float units."""

    logits_accumulator: np.ndarray
    logits: np.ndarray
    predicted_class: np.ndarray


class SpikingformerAcceleratorModel:
    """Full trained Spikingformer-4-384 inference graph on NeuraEdge primitives."""

    def __init__(
        self,
        archive: SpikingformerParameterArchive,
        driver: ProgramExecutor,
        *,
        layer_timeout_seconds: float = 120.0,
    ) -> None:
        if layer_timeout_seconds <= 0:
            raise ValueError("layer_timeout_seconds must be positive")
        self.archive = archive
        self.driver = driver
        self.projections = SpikingformerProjectionExecutor(archive, driver)
        self.layer_timeout_seconds = layer_timeout_seconds
        fixed = archive.manifest.get("fixed_point")
        if not isinstance(fixed, dict):
            raise ValueError("parameter archive has no fixed-point contract")
        self.fractional_bits = int(fixed["activation_fractional_bits"])
        self.fixed_scale = 1 << self.fractional_bits
        self.image_scale = float(fixed["image_input_scale"])
        self.image_zero_point = int(fixed["image_input_zero_point"])
        self.head_activation_scale = float(fixed["head_activation_scale"])
        required = {
            *(f"patch_embed.block{index}" for index in range(5)),
            *(f"block.{block}.attn.{name}"
              for block in range(4) for name in ("q", "k", "v", "proj")),
            *(f"block.{block}.mlp.fc{layer}"
              for block in range(4) for layer in (1, 2)),
        }
        missing = required - set(archive.projection_names)
        if missing:
            raise ValueError(f"parameter archive is missing projections: {sorted(missing)}")

    def _lif(self, value: np.ndarray, *, threshold: float = 1.0) -> np.ndarray:
        return lif_sequence_on_accelerator(
            self.driver,
            value,
            threshold_q10=int(round(threshold * self.fixed_scale)),
            leak_shift=1,
            reset_q10=0,
            timeout_seconds=self.layer_timeout_seconds,
        )

    def _project_tokens(self, name: str, spikes: np.ndarray) -> np.ndarray:
        if spikes.ndim != 4:
            raise ValueError("token projection input must have shape [T,B,N,C]")
        timesteps, batch, tokens, channels = spikes.shape
        rows = spikes.reshape(timesteps * batch * tokens, channels)
        result = self.projections.matrix(
            name, rows, timeout_seconds=self.layer_timeout_seconds)
        return result.reshape(timesteps, batch, tokens, -1)

    def _dynamic_binary_gemm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        program = compile_spike_gemm(left, right)
        return program.execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)

    def _dynamic_u8_gemm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        program = compile_u8_gemm(left, right)
        return program.execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)

    def _attention(self, block_index: int, value: np.ndarray) -> np.ndarray:
        timesteps, batch, channels, height, width = value.shape
        tokens = height * width
        projection_spikes = self._lif(value)
        token_spikes = projection_spikes.reshape(
            timesteps, batch, channels, tokens).transpose(0, 1, 3, 2)
        prefix = f"block.{block_index}.attn"
        query = self._lif(self._project_tokens(f"{prefix}.q", token_spikes))
        key = self._lif(self._project_tokens(f"{prefix}.k", token_spikes))
        projected_value = self._lif(self._project_tokens(f"{prefix}.v", token_spikes))

        heads = 12
        if channels % heads:
            raise ValueError("attention channels must divide into 12 heads")
        head_dim = channels // heads
        query = query.reshape(timesteps, batch, tokens, heads, head_dim).transpose(0, 1, 3, 2, 4)
        key = key.reshape(timesteps, batch, tokens, heads, head_dim).transpose(0, 1, 3, 2, 4)
        projected_value = projected_value.reshape(
            timesteps, batch, tokens, heads, head_dim).transpose(0, 1, 3, 2, 4)
        attention_value = np.empty(
            (timesteps, batch, heads, tokens, head_dim), dtype=np.int64)
        for timestep in range(timesteps):
            for batch_index in range(batch):
                for head in range(heads):
                    scores = self._dynamic_binary_gemm(
                        query[timestep, batch_index, head],
                        key[timestep, batch_index, head].T,
                    )
                    if scores.min() < 0 or scores.max() > 255:
                        raise OverflowError("binary QK score does not fit the dynamic-u8 AV path")
                    attention_value[timestep, batch_index, head] = self._dynamic_u8_gemm(
                        scores.astype(np.uint8),
                        projected_value[timestep, batch_index, head].astype(np.uint8),
                    )
        attention_q10 = attention_value * (self.fixed_scale // 8)
        merged = attention_q10.transpose(0, 1, 2, 4, 3).reshape(
            timesteps, batch, channels, tokens)
        attention_spikes = self._lif(merged, threshold=0.5)
        output_rows = attention_spikes.transpose(0, 1, 3, 2).reshape(
            timesteps * batch * tokens, channels)
        output = self.projections.matrix(
            f"{prefix}.proj", output_rows,
            timeout_seconds=self.layer_timeout_seconds)
        return output.reshape(timesteps, batch, tokens, channels).transpose(
            0, 1, 3, 2).reshape(timesteps, batch, channels, height, width)

    def _mlp(self, block_index: int, value: np.ndarray) -> np.ndarray:
        timesteps, batch, channels, height, width = value.shape
        tokens = height * width
        first_spikes = self._lif(value)
        first_rows = first_spikes.reshape(
            timesteps, batch, channels, tokens).transpose(0, 1, 3, 2).reshape(
                timesteps * batch * tokens, channels)
        hidden = self.projections.matrix(
            f"block.{block_index}.mlp.fc1", first_rows,
            timeout_seconds=self.layer_timeout_seconds)
        hidden_channels = hidden.shape[1]
        hidden = hidden.reshape(timesteps, batch, tokens, hidden_channels).transpose(0, 1, 3, 2)
        hidden_spikes = self._lif(hidden)
        second_rows = hidden_spikes.transpose(0, 1, 3, 2).reshape(
            timesteps * batch * tokens, hidden_channels)
        output = self.projections.matrix(
            f"block.{block_index}.mlp.fc2", second_rows,
            timeout_seconds=self.layer_timeout_seconds)
        return output.reshape(timesteps, batch, tokens, channels).transpose(
            0, 1, 3, 2).reshape(timesteps, batch, channels, height, width)

    def _tokenizer(self, image_u8: np.ndarray) -> np.ndarray:
        batch = image_u8.shape[0]
        first = self.projections.conv2d(
            "patch_embed.block0",
            image_u8,
            activation_zero_point=self.image_zero_point,
            padding=1,
            timeout_seconds=self.layer_timeout_seconds,
        )
        value = np.repeat(first[np.newaxis], 4, axis=0)
        for index in (1, 2):
            spikes = self._lif(value)
            flat = spikes.reshape(-1, *spikes.shape[2:])
            projected = self.projections.conv2d(
                f"patch_embed.block{index}",
                flat,
                padding=1,
                timeout_seconds=self.layer_timeout_seconds,
            )
            value = projected.reshape(4, batch, *projected.shape[1:])
        for index in (3, 4):
            spikes = self._lif(value)
            flat = spikes.reshape(-1, *spikes.shape[2:])
            pooled = binary_max_pool2d_on_accelerator(
                self.driver,
                flat,
                timeout_seconds=self.layer_timeout_seconds,
            )
            projected = self.projections.conv2d(
                f"patch_embed.block{index}",
                pooled,
                padding=1,
                timeout_seconds=self.layer_timeout_seconds,
            )
            value = projected.reshape(4, batch, *projected.shape[1:])
        return value

    def infer(self, normalized_image: np.ndarray) -> SpikingformerInferenceResult:
        """Run normalized CIFAR-10 NCHW images through the complete trained graph."""

        image = np.asarray(normalized_image, dtype=np.float32)
        if image.ndim != 4 or image.shape[1:] != (3, 32, 32) or image.shape[0] == 0:
            raise ValueError("input must have shape [batch,3,32,32]")
        image_u8 = np.rint(image / self.image_scale + self.image_zero_point)
        image_u8 = np.clip(image_u8, 0, 255).astype(np.uint8)
        self.driver.execute_program((ProgramRecord(clear_status_command()),),
                                    timeout_seconds=self.layer_timeout_seconds)
        value = self._tokenizer(image_u8)
        for block_index in range(4):
            attention = self._attention(block_index, value)
            value = vector_binary_on_accelerator(
                self.driver, VectorOperation.ADD_SAT, value, attention,
                timeout_seconds=self.layer_timeout_seconds)
            mlp = self._mlp(block_index, value)
            value = vector_binary_on_accelerator(
                self.driver, VectorOperation.ADD_SAT, value, mlp,
                timeout_seconds=self.layer_timeout_seconds)

        pooled_sum = global_spatiotemporal_sum_on_accelerator(
            self.driver,
            value,
            timeout_seconds=self.layer_timeout_seconds,
        )
        divisor = value.shape[0] * value.shape[3] * value.shape[4]
        denominator = divisor * self.fixed_scale * self.head_activation_scale
        integer_denominator = int(round(denominator))
        if (not math.isclose(denominator, integer_denominator, rel_tol=0.0, abs_tol=1e-9)
                or integer_denominator <= 0
                or integer_denominator & (integer_denominator - 1)):
            raise ValueError("head pooling denominator must be an exact positive power of two")
        head_integer = power_of_two_requantize_on_accelerator(
            self.driver,
            pooled_sum,
            division_shift=integer_denominator.bit_length() - 1,
            clamp_minimum=-128,
            clamp_maximum=127,
            timeout_seconds=self.layer_timeout_seconds,
        ).astype(np.int16)
        head = self.archive.head
        head_program = compile_u8_gemm(
            (head_integer + 128).astype(np.uint8),
            self.projections._encoded_dense_weights(head.weight_k_by_n),
            activation_zero_point=128,
            weight_zero_point=128,
            bias=head.bias_accumulator_units,
        )
        logits_accumulator = head_program.execute_batched(
            self.driver, timeout_seconds=self.layer_timeout_seconds)
        head_manifest = self.archive.manifest["head"]
        accumulator_scale = float(head_manifest["accumulator_scale"])
        logits = logits_accumulator.astype(np.float64) * accumulator_scale
        return SpikingformerInferenceResult(
            logits_accumulator=logits_accumulator,
            logits=logits,
            predicted_class=np.argmax(logits_accumulator, axis=1),
        )
