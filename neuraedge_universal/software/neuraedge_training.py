"""Exact tiled backward compiler for the NeuraEdge training datapath.

For ``Y = XW`` the accelerator evaluates both matrix derivatives:
``dX = dY W^T`` and ``dW = X^T dY``.  Hardware retains 47-bit unrounded
partial sums between commands, so this compiler may tile every dimension
without changing the integer equation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from neuraedge_protocol import (
    LinearGradientMode,
    LutSuboperation,
    ProgramRecord,
    StreamBeat,
    linear_backward_command,
    lut_command,
    pack_linear_backward_beats,
    pack_lut_eval_beat,
    pack_lut_load_beats,
    pack_optimizer_beats,
    pack_vector_beats,
    sgd_command,
    unpack_linear_backward_result,
    unpack_lut_result,
    unpack_optimizer_result,
    unpack_vector_result,
    vector_command,
)
from neuraedge_golden import VectorOperation, round_shift_signed, saturate_signed


TILE = 8


class TrainingCommandExecutor(Protocol):
    def execute(
        self,
        command,
        input_beats: Sequence[StreamBeat] = (),
        *,
        output_beats: int = 0,
        timeout_seconds: float = 5.0,
    ) -> tuple[int, ...]: ...


class TrainingProgramExecutor(TrainingCommandExecutor, Protocol):
    def execute_program(
        self,
        records: Sequence[ProgramRecord],
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class BackwardInstruction:
    """One partial tile and the global matrix region to which it belongs."""

    record: ProgramRecord
    mode: LinearGradientMode
    row_start: int
    input_start: int
    output_start: int
    rows: int
    inputs: int
    outputs: int
    accumulate: bool
    suppress_output: bool


@dataclass(frozen=True)
class CompiledLinearBackward:
    """Executable descriptor program for one arbitrary signed linear layer."""

    shape: tuple[int, int, int]
    input_gradient_shift: int
    weight_gradient_shift: int
    instructions: tuple[BackwardInstruction, ...]
    estimated_input_bytes: int
    mathematical_multiplications: int

    @property
    def records(self) -> tuple[ProgramRecord, ...]:
        return tuple(instruction.record for instruction in self.instructions)

    @property
    def output_beats(self) -> int:
        return sum(instruction.record.output_beats for instruction in self.instructions)

    @property
    def partial_tile_count(self) -> int:
        return len(self.instructions)

    def _decode_outputs(self, instruction_words: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
        rows, inputs, outputs = self.shape
        input_gradient = np.zeros((rows, inputs), dtype=np.int64)
        weight_gradient = np.zeros((inputs, outputs), dtype=np.int64)
        if len(instruction_words) != len(self.instructions):
            raise RuntimeError("backward result partition does not match the instruction count")
        for instruction, words in zip(self.instructions, instruction_words):
            if len(words) != instruction.record.output_beats:
                raise RuntimeError(
                    f"backward tile returned {len(words)} beats; "
                    f"expected {instruction.record.output_beats}"
                )
            if instruction.suppress_output:
                continue
            dx_tile, dw_tile = unpack_linear_backward_result(
                words,
                instruction.rows,
                instruction.inputs,
                instruction.outputs,
                mode=instruction.mode,
            )
            if instruction.mode == LinearGradientMode.INPUT:
                input_gradient[
                    instruction.row_start:instruction.row_start + instruction.rows,
                    instruction.input_start:instruction.input_start + instruction.inputs,
                ] = np.asarray(dx_tile, dtype=np.int64)
            else:
                weight_gradient[
                    instruction.input_start:instruction.input_start + instruction.inputs,
                    instruction.output_start:instruction.output_start + instruction.outputs,
                ] = np.asarray(dw_tile, dtype=np.int64)
        return input_gradient, weight_gradient

    def execute(
        self,
        driver: TrainingCommandExecutor,
        *,
        timeout_seconds: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        partitions: list[tuple[int, ...]] = []
        for instruction in self.instructions:
            partitions.append(driver.execute(
                instruction.record.command,
                instruction.record.input_beats,
                output_beats=instruction.record.output_beats,
                timeout_seconds=timeout_seconds,
            ))
        return self._decode_outputs(partitions)

    def execute_batched(
        self,
        driver: TrainingProgramExecutor,
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        words = driver.execute_program(self.records, timeout_seconds=timeout_seconds)
        if len(words) != self.output_beats:
            raise RuntimeError(
                f"backward program returned {len(words)} beats; expected {self.output_beats}"
            )
        partitions: list[tuple[int, ...]] = []
        cursor = 0
        for instruction in self.instructions:
            end = cursor + instruction.record.output_beats
            partitions.append(tuple(words[cursor:end]))
            cursor = end
        return self._decode_outputs(partitions)


@dataclass(frozen=True)
class CompiledAffineBackward:
    """Linear backward plan whose final augmented row represents the bias."""

    linear: CompiledLinearBackward
    input_features: int

    def split(
        self,
        augmented_input_gradient: np.ndarray,
        augmented_weight_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            augmented_input_gradient[:, :self.input_features],
            augmented_weight_gradient[:self.input_features, :],
            augmented_weight_gradient[self.input_features, :],
        )

    def execute(
        self,
        driver: TrainingCommandExecutor,
        *,
        timeout_seconds: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.split(*self.linear.execute(driver, timeout_seconds=timeout_seconds))

    def execute_batched(
        self,
        driver: TrainingProgramExecutor,
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.split(*self.linear.execute_batched(driver, timeout_seconds=timeout_seconds))


@dataclass(frozen=True)
class ConvBackwardGroup:
    input_channel_start: int
    output_channel_start: int
    input_channels: int
    output_channels: int
    affine: CompiledAffineBackward


@dataclass(frozen=True)
class CompiledConv2dBackward:
    input_shape: tuple[int, int, int, int]
    weight_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    group_programs: tuple[ConvBackwardGroup, ...]

    @property
    def mathematical_multiplications(self) -> int:
        return sum(
            group.affine.linear.mathematical_multiplications
            for group in self.group_programs
        )

    @property
    def partial_tile_count(self) -> int:
        return sum(
            group.affine.linear.partial_tile_count for group in self.group_programs
        )

    def execute_batched(
        self,
        driver: TrainingProgramExecutor,
        *,
        timeout_seconds: float = 30.0,
        fabric_col2im: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_gradient = np.zeros(self.input_shape, dtype=np.int64)
        weight_gradient = np.zeros(self.weight_shape, dtype=np.int64)
        bias_gradient = np.zeros(self.output_shape[1], dtype=np.int64)
        output_height, output_width = self.output_shape[2:]
        kernel_h, kernel_w = self.kernel
        for group in self.group_programs:
            patch_gradient, matrix_gradient, group_bias = group.affine.execute_batched(
                driver, timeout_seconds=timeout_seconds
            )
            if fabric_col2im:
                group_input_gradient = execute_col2im_accumulate(
                    driver,
                    patch_gradient,
                    batch=self.input_shape[0],
                    channels=group.input_channels,
                    input_height=self.input_shape[2],
                    input_width=self.input_shape[3],
                    output_height=output_height,
                    output_width=output_width,
                    kernel=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    timeout_seconds=timeout_seconds,
                )
            else:
                group_input_gradient = col2im_accumulate_fixed(
                    patch_gradient,
                    batch=self.input_shape[0],
                    channels=group.input_channels,
                    input_height=self.input_shape[2],
                    input_width=self.input_shape[3],
                    output_height=output_height,
                    output_width=output_width,
                    kernel=self.kernel,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
            input_start = group.input_channel_start
            output_start = group.output_channel_start
            input_gradient[:, input_start:input_start + group.input_channels] = group_input_gradient
            matrix = matrix_gradient.T.reshape(
                group.output_channels,
                group.input_channels,
                kernel_h,
                kernel_w,
            )
            weight_gradient[
                output_start:output_start + group.output_channels
            ] = matrix
            bias_gradient[
                output_start:output_start + group.output_channels
            ] = group_bias
        return input_gradient, weight_gradient, bias_gradient


def _signed_matrix(name: str, value, bits: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    result = array.astype(np.int64)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if np.any(result < minimum) or np.any(result > maximum):
        raise ValueError(f"{name} must fit signed {bits}-bit values")
    return result


def _conv_output_size(
    input_size: int,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    return (input_size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def im2col_signed_nchw(
    values,
    *,
    kernel,
    stride,
    padding,
    dilation,
) -> np.ndarray:
    """Flatten NCHW convolution patches in batch/output-y/output-x order."""

    source = np.asarray(values, dtype=np.int64)
    if source.ndim != 4:
        raise ValueError("im2col source must be an NCHW tensor")
    kernel_h, kernel_w = _pair("kernel", kernel)
    stride_h, stride_w = _pair("stride", stride)
    dilation_h, dilation_w = _pair("dilation", dilation)
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
    batch, channels, height, width = source.shape
    output_height = _conv_output_size(height, kernel_h, stride_h, pad_h, dilation_h)
    output_width = _conv_output_size(width, kernel_w, stride_w, pad_w, dilation_w)
    patches = np.zeros(
        (batch * output_height * output_width, channels * kernel_h * kernel_w),
        dtype=np.int64,
    )
    row = 0
    for batch_index in range(batch):
        for output_y in range(output_height):
            for output_x in range(output_width):
                column = 0
                for channel in range(channels):
                    for kernel_y in range(kernel_h):
                        input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                        for kernel_x in range(kernel_w):
                            input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                            if 0 <= input_y < height and 0 <= input_x < width:
                                patches[row, column] = source[
                                    batch_index, channel, input_y, input_x
                                ]
                            column += 1
                row += 1
    return patches


def col2im_accumulate_fixed(
    patch_gradients,
    *,
    batch: int,
    channels: int,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    kernel,
    stride,
    padding,
    dilation,
) -> np.ndarray:
    """Scatter patch gradients with one signed-s32 saturation per addition."""

    patches = np.asarray(patch_gradients, dtype=np.int64)
    kernel_h, kernel_w = _pair("kernel", kernel)
    stride_h, stride_w = _pair("stride", stride)
    dilation_h, dilation_w = _pair("dilation", dilation)
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
    expected_shape = (
        batch * output_height * output_width,
        channels * kernel_h * kernel_w,
    )
    if patches.shape != expected_shape:
        raise ValueError(f"patch gradient shape must be {expected_shape}")
    output = np.zeros((batch, channels, input_height, input_width), dtype=np.int64)
    row = 0
    for batch_index in range(batch):
        for output_y in range(output_height):
            for output_x in range(output_width):
                column = 0
                for channel in range(channels):
                    for kernel_y in range(kernel_h):
                        input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                        for kernel_x in range(kernel_w):
                            input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                            if 0 <= input_y < input_height and 0 <= input_x < input_width:
                                value = output[batch_index, channel, input_y, input_x]
                                output[batch_index, channel, input_y, input_x] = saturate_signed(
                                    np.asarray(value + patches[row, column], dtype=np.int64), 32
                                )
                            column += 1
                row += 1
    return output


def execute_col2im_accumulate(
    driver: TrainingCommandExecutor,
    patch_gradients,
    *,
    batch: int,
    channels: int,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    kernel,
    stride,
    padding,
    dilation,
    timeout_seconds: float = 5.0,
) -> np.ndarray:
    """Scatter every overlapping convolution gradient with fabric vector adds."""

    patches = np.asarray(patch_gradients, dtype=np.int64)
    kernel_h, kernel_w = _pair("kernel", kernel)
    stride_h, stride_w = _pair("stride", stride)
    dilation_h, dilation_w = _pair("dilation", dilation)
    pad_h, pad_w = (padding, padding) if isinstance(padding, int) else tuple(padding)
    expected_shape = (
        batch * output_height * output_width,
        channels * kernel_h * kernel_w,
    )
    if patches.shape != expected_shape:
        raise ValueError(f"patch gradient shape must be {expected_shape}")
    output = np.zeros((batch, channels, input_height, input_width), dtype=np.int64)
    row = 0
    pending: list[tuple[tuple[int, int, int, int], int]] = []

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        left = [int(output[address]) for address, _ in pending]
        right = [value for _, value in pending]
        left.extend([0] * (8 - len(pending)))
        right.extend([0] * (8 - len(pending)))
        summed = _execute_vector8(
            driver,
            VectorOperation.ADD_SAT,
            left,
            right,
            timeout_seconds=timeout_seconds,
        )
        for lane, (address, _) in enumerate(pending):
            output[address] = summed[lane]
        pending = []

    for batch_index in range(batch):
        for output_y in range(output_height):
            for output_x in range(output_width):
                column = 0
                for channel in range(channels):
                    for kernel_y in range(kernel_h):
                        input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                        for kernel_x in range(kernel_w):
                            input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                            if 0 <= input_y < input_height and 0 <= input_x < input_width:
                                address = (batch_index, channel, input_y, input_x)
                                if len(pending) == 8 or any(item[0] == address for item in pending):
                                    flush()
                                pending.append((address, int(patches[row, column])))
                            column += 1
                row += 1
    flush()
    return output


def _tile_starts(size: int) -> range:
    return range(0, size, TILE)


def compile_linear_backward(
    activations,
    weights,
    output_gradients,
    *,
    input_gradient_shift: int = 0,
    weight_gradient_shift: int = 0,
) -> CompiledLinearBackward:
    """Compile exact dX and dW passes for matrices of any positive shape."""

    x = _signed_matrix("activations", activations, 16)
    w = _signed_matrix("weights", weights, 16)
    dy = _signed_matrix("output_gradients", output_gradients, 16)
    rows, inputs = x.shape
    weight_inputs, outputs = w.shape
    if weight_inputs != inputs or dy.shape != (rows, outputs):
        raise ValueError("linear backward matrix dimensions do not align")
    if not 0 <= input_gradient_shift <= 31:
        raise ValueError("input_gradient_shift must be in [0,31]")
    if not 0 <= weight_gradient_shift <= 31:
        raise ValueError("weight_gradient_shift must be in [0,31]")

    instructions: list[BackwardInstruction] = []
    input_bytes = 0

    # dX tiles retain one MxK accumulator while traversing every N chunk.
    for row_start in _tile_starts(rows):
        row_end = min(row_start + TILE, rows)
        for input_start in _tile_starts(inputs):
            input_end = min(input_start + TILE, inputs)
            output_starts = tuple(_tile_starts(outputs))
            for chunk_index, output_start in enumerate(output_starts):
                output_end = min(output_start + TILE, outputs)
                final = chunk_index == len(output_starts) - 1
                x_tile = x[row_start:row_end, input_start:input_end]
                w_tile = w[input_start:input_end, output_start:output_end]
                dy_tile = dy[row_start:row_end, output_start:output_end]
                beats = pack_linear_backward_beats(
                    x_tile.tolist(),
                    w_tile.tolist(),
                    dy_tile.tolist(),
                    mode=LinearGradientMode.INPUT,
                )
                record = ProgramRecord(
                    linear_backward_command(
                        row_end - row_start,
                        input_end - input_start,
                        output_end - output_start,
                        input_gradient_shift=input_gradient_shift,
                        weight_gradient_shift=weight_gradient_shift,
                        mode=LinearGradientMode.INPUT,
                        accumulate=chunk_index != 0,
                        suppress_output=not final,
                    ),
                    beats,
                    4 * (row_end - row_start) if final else 0,
                )
                instructions.append(BackwardInstruction(
                    record,
                    LinearGradientMode.INPUT,
                    row_start,
                    input_start,
                    output_start,
                    row_end - row_start,
                    input_end - input_start,
                    output_end - output_start,
                    chunk_index != 0,
                    not final,
                ))
                input_bytes += len(beats) * 8

    # dW tiles retain one KxN accumulator while traversing every M chunk.
    for input_start in _tile_starts(inputs):
        input_end = min(input_start + TILE, inputs)
        for output_start in _tile_starts(outputs):
            output_end = min(output_start + TILE, outputs)
            row_starts = tuple(_tile_starts(rows))
            for chunk_index, row_start in enumerate(row_starts):
                row_end = min(row_start + TILE, rows)
                final = chunk_index == len(row_starts) - 1
                x_tile = x[row_start:row_end, input_start:input_end]
                w_tile = w[input_start:input_end, output_start:output_end]
                dy_tile = dy[row_start:row_end, output_start:output_end]
                beats = pack_linear_backward_beats(
                    x_tile.tolist(),
                    w_tile.tolist(),
                    dy_tile.tolist(),
                    mode=LinearGradientMode.WEIGHT,
                )
                record = ProgramRecord(
                    linear_backward_command(
                        row_end - row_start,
                        input_end - input_start,
                        output_end - output_start,
                        input_gradient_shift=input_gradient_shift,
                        weight_gradient_shift=weight_gradient_shift,
                        mode=LinearGradientMode.WEIGHT,
                        accumulate=chunk_index != 0,
                        suppress_output=not final,
                    ),
                    beats,
                    3 * (input_end - input_start) if final else 0,
                )
                instructions.append(BackwardInstruction(
                    record,
                    LinearGradientMode.WEIGHT,
                    row_start,
                    input_start,
                    output_start,
                    row_end - row_start,
                    input_end - input_start,
                    output_end - output_start,
                    chunk_index != 0,
                    not final,
                ))
                input_bytes += len(beats) * 8

    return CompiledLinearBackward(
        (rows, inputs, outputs),
        input_gradient_shift,
        weight_gradient_shift,
        tuple(instructions),
        input_bytes,
        2 * rows * inputs * outputs,
    )


def compile_affine_backward(
    activations,
    weights,
    bias,
    output_gradients,
    *,
    input_gradient_shift: int = 0,
    weight_gradient_shift: int = 0,
) -> CompiledAffineBackward:
    """Compile weight, bias, and input gradients for ``Y = XW + b``."""

    x = _signed_matrix("activations", activations, 16)
    w = _signed_matrix("weights", weights, 16)
    bias_array = np.asarray(bias)
    if bias_array.ndim != 1 or bias_array.shape[0] != w.shape[1]:
        raise ValueError("bias must be one-dimensional with one value per output")
    bias_matrix = _signed_matrix("bias", bias_array.reshape(1, -1), 16)
    if x.shape[1] != w.shape[0]:
        raise ValueError("affine activation and weight dimensions do not align")
    augmented_x = np.concatenate(
        (x, np.ones((x.shape[0], 1), dtype=np.int64)), axis=1
    )
    augmented_w = np.concatenate((w, bias_matrix), axis=0)
    return CompiledAffineBackward(
        compile_linear_backward(
            augmented_x,
            augmented_w,
            output_gradients,
            input_gradient_shift=input_gradient_shift,
            weight_gradient_shift=weight_gradient_shift,
        ),
        x.shape[1],
    )


def compile_conv2d_backward(
    activations,
    weights,
    bias,
    output_gradients,
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    input_gradient_shift: int = 0,
    weight_gradient_shift: int = 0,
) -> CompiledConv2dBackward:
    """Lower a signed NCHW convolution backward pass to exact affine tiles."""

    x = np.asarray(activations)
    w = np.asarray(weights)
    dy = np.asarray(output_gradients)
    bias_array = np.asarray(bias)
    for name, array, bits in (
        ("activations", x, 16),
        ("weights", w, 16),
        ("output_gradients", dy, 16),
        ("bias", bias_array, 16),
    ):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must contain integers")
        minimum = -(1 << (bits - 1))
        maximum = (1 << (bits - 1)) - 1
        if np.any(array < minimum) or np.any(array > maximum):
            raise ValueError(f"{name} must fit signed {bits}-bit values")
    if x.ndim != 4 or w.ndim != 4 or dy.ndim != 4 or bias_array.ndim != 1:
        raise ValueError("convolution operands must use NCHW/OIHW tensors and a bias vector")
    if groups <= 0:
        raise ValueError("groups must be positive")
    batch, input_channels, input_height, input_width = x.shape
    output_channels, weight_inputs, kernel_h, kernel_w = w.shape
    if input_channels % groups or output_channels % groups:
        raise ValueError("input and output channels must be divisible by groups")
    input_channels_per_group = input_channels // groups
    output_channels_per_group = output_channels // groups
    if weight_inputs != input_channels_per_group:
        raise ValueError("weight input channels do not match the grouped convolution")
    if bias_array.shape != (output_channels,):
        raise ValueError("bias must contain one value per output channel")
    stride_pair = _pair("stride", stride)
    dilation_pair = _pair("dilation", dilation)
    if isinstance(padding, int):
        padding_pair = (int(padding), int(padding))
    else:
        padding_pair = tuple(map(int, padding))
        if len(padding_pair) != 2:
            raise ValueError("padding must be an integer or a pair")
    if min(padding_pair) < 0:
        raise ValueError("padding must be nonnegative")
    output_height = _conv_output_size(
        input_height, kernel_h, stride_pair[0], padding_pair[0], dilation_pair[0]
    )
    output_width = _conv_output_size(
        input_width, kernel_w, stride_pair[1], padding_pair[1], dilation_pair[1]
    )
    expected_output_shape = (batch, output_channels, output_height, output_width)
    if dy.shape != expected_output_shape:
        raise ValueError(
            f"output_gradients must have shape {expected_output_shape}; received {dy.shape}"
        )
    programs: list[ConvBackwardGroup] = []
    for group in range(groups):
        input_start = group * input_channels_per_group
        output_start = group * output_channels_per_group
        patches = im2col_signed_nchw(
            x[:, input_start:input_start + input_channels_per_group],
            kernel=(kernel_h, kernel_w),
            stride=stride_pair,
            padding=padding_pair,
            dilation=dilation_pair,
        )
        weight_matrix = w[
            output_start:output_start + output_channels_per_group
        ].reshape(output_channels_per_group, -1).T
        gradient_matrix = dy[
            :, output_start:output_start + output_channels_per_group
        ].transpose(0, 2, 3, 1).reshape(-1, output_channels_per_group)
        programs.append(ConvBackwardGroup(
            input_start,
            output_start,
            input_channels_per_group,
            output_channels_per_group,
            compile_affine_backward(
                patches,
                weight_matrix,
                bias_array[output_start:output_start + output_channels_per_group],
                gradient_matrix,
                input_gradient_shift=input_gradient_shift,
                weight_gradient_shift=weight_gradient_shift,
            ),
        ))
    return CompiledConv2dBackward(
        tuple(map(int, x.shape)),
        tuple(map(int, w.shape)),
        tuple(map(int, expected_output_shape)),
        (kernel_h, kernel_w),
        stride_pair,
        padding_pair,
        dilation_pair,
        groups,
        tuple(programs),
    )


def sigmoid_surrogate_lut_q15(
    *,
    alpha: float = 4.0,
    membrane_fractional_bits: int = 10,
    address_shift: int | None = None,
) -> tuple[np.ndarray, int]:
    """Quantize SpikingJelly's sigmoid surrogate derivative into 256 Q15 entries."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("surrogate alpha must be finite and positive")
    if not 0 <= membrane_fractional_bits <= 22:
        raise ValueError("membrane_fractional_bits must be in [0,22]")
    selected_shift = (
        max(membrane_fractional_bits - 5, 0)
        if address_shift is None
        else int(address_shift)
    )
    if not 0 <= selected_shift <= membrane_fractional_bits:
        raise ValueError("address_shift must be in [0, membrane_fractional_bits]")
    addresses = np.arange(256, dtype=np.float64)
    delta = (addresses - 128.0) * (2.0 ** selected_shift)
    delta /= 2.0 ** membrane_fractional_bits
    sigmoid = 1.0 / (1.0 + np.exp(-alpha * delta))
    derivative = alpha * sigmoid * (1.0 - sigmoid)
    table = np.rint(derivative * (1 << 15))
    table = np.clip(table, 0, (1 << 15) - 1).astype(np.int64)
    return table, selected_shift


def surrogate_lut_addresses(
    pre_reset_membrane,
    threshold: int,
    address_shift: int,
) -> np.ndarray:
    """Map signed fixed-point membrane distance to a saturating LUT address."""

    if not 0 <= address_shift <= 31:
        raise ValueError("address_shift must be in [0,31]")
    membrane = np.asarray(pre_reset_membrane, dtype=np.int64)
    distance = membrane - int(threshold)
    quantized = round_shift_signed(distance, address_shift)
    return np.clip(quantized + 128, 0, 255).astype(np.uint8)


def lif_bptt_fixed(
    pre_reset_membrane,
    spikes,
    spike_gradients,
    *,
    threshold: int,
    membrane_gradients=None,
    reciprocal_tau_q15: int = 1 << 14,
    surrogate_table_q15=None,
    surrogate_address_shift: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact detached-reset hard-LIF BPTT using signed Q15 products."""

    membrane = np.asarray(pre_reset_membrane, dtype=np.int64)
    spike = np.asarray(spikes, dtype=np.int64)
    grad_spike = np.asarray(spike_gradients, dtype=np.int64)
    if membrane.ndim != 2 or spike.shape != membrane.shape or grad_spike.shape != membrane.shape:
        raise ValueError("LIF trace, spike, and gradient arrays must share shape [time, neurons]")
    if np.any((spike != 0) & (spike != 1)):
        raise ValueError("LIF spikes must be binary")
    if membrane_gradients is None:
        grad_v = np.zeros_like(membrane)
    else:
        grad_v = np.asarray(membrane_gradients, dtype=np.int64)
        if grad_v.shape != membrane.shape:
            raise ValueError("membrane gradients must match the LIF trace shape")
    if not 0 <= reciprocal_tau_q15 <= (1 << 15):
        raise ValueError("reciprocal_tau_q15 must be in [0,32768]")
    if surrogate_table_q15 is None:
        surrogate_table_q15, _ = sigmoid_surrogate_lut_q15(
            address_shift=surrogate_address_shift
        )
    table = np.asarray(surrogate_table_q15, dtype=np.int64)
    if table.shape != (256,) or np.any(table < 0) or np.any(table > 32767):
        raise ValueError("surrogate table must contain 256 nonnegative signed-Q15 values")
    addresses = surrogate_lut_addresses(membrane, threshold, surrogate_address_shift)
    derivative = table[addresses]
    grad_input = np.zeros_like(membrane)
    grad_h_trace = np.zeros_like(membrane)
    grad_h_next = np.zeros(membrane.shape[1], dtype=np.int64)
    leak_q15 = (1 << 15) - reciprocal_tau_q15
    for timestep in range(membrane.shape[0] - 1, -1, -1):
        temporal = saturate_signed(
            round_shift_signed(grad_h_next * leak_q15, 15), 32
        )
        membrane_path = saturate_signed(grad_v[timestep] + temporal, 32)
        membrane_path = np.where(spike[timestep] == 0, membrane_path, 0)
        spike_path = saturate_signed(
            round_shift_signed(grad_spike[timestep] * derivative[timestep], 15), 32
        )
        grad_h = saturate_signed(membrane_path + spike_path, 32)
        grad_input[timestep] = saturate_signed(
            round_shift_signed(grad_h * reciprocal_tau_q15, 15), 32
        )
        grad_h_trace[timestep] = grad_h
        grad_h_next = grad_h
    return grad_input, grad_h_trace


def _execute_vector8(
    driver: TrainingCommandExecutor,
    operation: VectorOperation,
    operand_a: Sequence[int],
    operand_b: Sequence[int],
    *,
    timeout_seconds: float,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
) -> np.ndarray:
    words = driver.execute(
        vector_command(int(operation)),
        pack_vector_beats(
            operand_a,
            operand_b,
            clamp_minimum=clamp_minimum,
            clamp_maximum=clamp_maximum,
        ),
        output_beats=4,
        timeout_seconds=timeout_seconds,
    )
    if len(words) != 4:
        raise RuntimeError(f"vector operation returned {len(words)} beats; expected four")
    return np.asarray(unpack_vector_result(words), dtype=np.int64)


def execute_sgdw_update(
    driver: TrainingCommandExecutor,
    weights,
    gradients,
    velocities,
    *,
    learning_rate_q16: int,
    momentum_q16: int,
    weight_decay_q16: int,
    update_right_shift: int = 16,
    timeout_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Update an arbitrary tensor through the fabric's eight SGDW lanes."""

    weight = np.asarray(weights, dtype=np.int64)
    gradient = np.asarray(gradients, dtype=np.int64)
    velocity = np.asarray(velocities, dtype=np.int64)
    if weight.shape != gradient.shape or weight.shape != velocity.shape or weight.size == 0:
        raise ValueError("weights, gradients, and velocities must share a non-empty shape")
    if np.any(weight < -(1 << 15)) or np.any(weight > (1 << 15) - 1):
        raise ValueError("optimizer weights must fit signed 16-bit")
    for name, value in (("gradients", gradient), ("velocities", velocity)):
        if np.any(value < -(1 << 23)) or np.any(value > (1 << 23) - 1):
            raise ValueError(f"optimizer {name} must fit signed 24-bit")
    flat_weight = weight.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    flat_velocity = velocity.reshape(-1)
    next_weight = np.zeros_like(flat_weight)
    next_velocity = np.zeros_like(flat_velocity)
    command = sgd_command(
        learning_rate_q16,
        momentum_q16,
        weight_decay_q16,
        update_right_shift=update_right_shift,
    )
    for start in range(0, flat_weight.size, 8):
        stop = min(start + 8, flat_weight.size)
        active = stop - start

        def padded(values):
            return tuple(map(int, values)) + (0,) * (8 - active)

        words = driver.execute(
            command,
            pack_optimizer_beats(
                padded(flat_weight[start:stop]),
                padded(flat_gradient[start:stop]),
                padded(flat_velocity[start:stop]),
            ),
            output_beats=8,
            timeout_seconds=timeout_seconds,
        )
        updated_weight, updated_velocity = unpack_optimizer_result(words)
        next_weight[start:stop] = updated_weight[:active]
        next_velocity[start:stop] = updated_velocity[:active]
    return next_weight.reshape(weight.shape), next_velocity.reshape(weight.shape)


def execute_int8_weight_projection(
    driver: TrainingCommandExecutor,
    master_weights,
    *,
    timeout_seconds: float = 5.0,
) -> np.ndarray:
    """Clamp signed-s16 master weights to the deployed signed-int8 tensor."""

    master = np.asarray(master_weights, dtype=np.int64)
    if master.size == 0 or np.any(master < -(1 << 15)) or np.any(master > (1 << 15) - 1):
        raise ValueError("master_weights must be a non-empty signed-16 tensor")
    flat = master.reshape(-1)
    result = np.zeros_like(flat)
    for start in range(0, flat.size, 8):
        stop = min(start + 8, flat.size)
        active = stop - start
        operand = tuple(map(int, flat[start:stop])) + (0,) * (8 - active)
        clamped = _execute_vector8(
            driver,
            VectorOperation.CLAMP,
            operand,
            (0,) * 8,
            timeout_seconds=timeout_seconds,
            clamp_minimum=-128,
            clamp_maximum=127,
        )
        result[start:stop] = clamped[:active]
    return result.reshape(master.shape).astype(np.int8)


def execute_lif_bptt(
    driver: TrainingCommandExecutor,
    pre_reset_membrane,
    spikes,
    spike_gradients,
    *,
    threshold: int,
    membrane_gradients=None,
    reciprocal_tau_q15: int = 1 << 14,
    surrogate_table_q15=None,
    surrogate_address_shift: int = 5,
    load_surrogate_table: bool = True,
    timeout_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute detached-reset hard-LIF BPTT entirely through fabric operators."""

    membrane = np.asarray(pre_reset_membrane, dtype=np.int64)
    spike = np.asarray(spikes, dtype=np.int64)
    grad_spike = np.asarray(spike_gradients, dtype=np.int64)
    if membrane.ndim != 2 or spike.shape != membrane.shape or grad_spike.shape != membrane.shape:
        raise ValueError("LIF trace, spike, and gradient arrays must share shape [time, neurons]")
    if membrane_gradients is None:
        grad_v = np.zeros_like(membrane)
    else:
        grad_v = np.asarray(membrane_gradients, dtype=np.int64)
        if grad_v.shape != membrane.shape:
            raise ValueError("membrane gradients must match the LIF trace shape")
    if surrogate_table_q15 is None:
        surrogate_table_q15, _ = sigmoid_surrogate_lut_q15(
            address_shift=surrogate_address_shift
        )
    table = np.asarray(surrogate_table_q15, dtype=np.int64)
    if table.shape != (256,):
        raise ValueError("surrogate table must contain exactly 256 entries")
    if load_surrogate_table:
        load_words = driver.execute(
            lut_command(LutSuboperation.LOAD, 256),
            pack_lut_load_beats(
                tuple((address, int(table[address])) for address in range(256))),
            output_beats=0,
            timeout_seconds=timeout_seconds,
        )
        if load_words:
            raise RuntimeError("LUT load unexpectedly returned stream data")

    addresses = surrogate_lut_addresses(membrane, threshold, surrogate_address_shift)
    time_steps, neurons = membrane.shape
    grad_input = np.zeros_like(membrane)
    grad_h_trace = np.zeros_like(membrane)
    grad_h_next = np.zeros(neurons, dtype=np.int64)
    leak_q15 = (1 << 15) - int(reciprocal_tau_q15)
    for timestep in range(time_steps - 1, -1, -1):
        for start in range(0, neurons, 8):
            stop = min(start + 8, neurons)
            active = stop - start

            def padded(values, fill=0):
                return tuple(map(int, values)) + (fill,) * (8 - active)

            temporal = _execute_vector8(
                driver,
                VectorOperation.MULTIPLY_Q15,
                padded(grad_h_next[start:stop]),
                (leak_q15,) * 8,
                timeout_seconds=timeout_seconds,
            )
            membrane_path = _execute_vector8(
                driver,
                VectorOperation.ADD_SAT,
                padded(grad_v[timestep, start:stop]),
                temporal,
                timeout_seconds=timeout_seconds,
            )
            membrane_path = _execute_vector8(
                driver,
                VectorOperation.SELECT_ZERO,
                padded(spike[timestep, start:stop]),
                membrane_path,
                timeout_seconds=timeout_seconds,
            )
            lut_words = driver.execute(
                lut_command(LutSuboperation.EVALUATE, 8),
                pack_lut_eval_beat(padded(addresses[timestep, start:stop], 128)),
                output_beats=2,
                timeout_seconds=timeout_seconds,
            )
            derivative = np.asarray(unpack_lut_result(lut_words), dtype=np.int64)
            spike_path = _execute_vector8(
                driver,
                VectorOperation.MULTIPLY_Q15,
                padded(grad_spike[timestep, start:stop]),
                derivative,
                timeout_seconds=timeout_seconds,
            )
            grad_h = _execute_vector8(
                driver,
                VectorOperation.ADD_SAT,
                membrane_path,
                spike_path,
                timeout_seconds=timeout_seconds,
            )
            input_chunk = _execute_vector8(
                driver,
                VectorOperation.MULTIPLY_Q15,
                grad_h,
                (int(reciprocal_tau_q15),) * 8,
                timeout_seconds=timeout_seconds,
            )
            grad_h_trace[timestep, start:stop] = grad_h[:active]
            grad_input[timestep, start:stop] = input_chunk[:active]
        grad_h_next = grad_h_trace[timestep].copy()
    return grad_input, grad_h_trace


def _pair(name: str, value) -> tuple[int, int]:
    if isinstance(value, int):
        result = (int(value), int(value))
    else:
        result = tuple(map(int, value))
        if len(result) != 2:
            raise ValueError(f"{name} must be an integer or a pair")
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"{name} values must be positive")
    return result


def maxpool2d_backward_fixed(
    values,
    output_gradients,
    *,
    kernel_size=3,
    stride=2,
    padding=1,
    dilation=1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point MaxPool2d backward with PyTorch's first-maximum tie rule."""

    source = np.asarray(values, dtype=np.int64)
    gradient = np.asarray(output_gradients, dtype=np.int64)
    if source.ndim != 4 or gradient.ndim != 4:
        raise ValueError("max-pool values and gradients must be NCHW tensors")
    kernel_h, kernel_w = _pair("kernel_size", kernel_size)
    stride_h, stride_w = _pair("stride", stride)
    dilation_h, dilation_w = _pair("dilation", dilation)
    if isinstance(padding, int):
        pad_h, pad_w = int(padding), int(padding)
    else:
        pad_h, pad_w = tuple(map(int, padding))
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding must be nonnegative")
    batch, channels, height, width = source.shape
    output_height = (
        height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_width = (
        width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if gradient.shape != (batch, channels, output_height, output_width):
        raise ValueError("max-pool output gradient shape does not match its geometry")
    input_gradient = np.zeros(source.shape, dtype=np.int64)
    winners = np.zeros(gradient.shape, dtype=np.int64)
    for batch_index in range(batch):
        for channel in range(channels):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    winner_value: int | None = None
                    winner_index = -1
                    for kernel_y in range(kernel_h):
                        input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                        if not 0 <= input_y < height:
                            continue
                        for kernel_x in range(kernel_w):
                            input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                            if not 0 <= input_x < width:
                                continue
                            candidate = int(source[batch_index, channel, input_y, input_x])
                            if winner_value is None or candidate > winner_value:
                                winner_value = candidate
                                winner_index = kernel_y * kernel_w + kernel_x
                    if winner_index < 0:
                        raise ValueError("a max-pool window contains no source elements")
                    winners[batch_index, channel, output_y, output_x] = winner_index
                    input_y = (
                        output_y * stride_h - pad_h
                        + (winner_index // kernel_w) * dilation_h
                    )
                    input_x = (
                        output_x * stride_w - pad_w
                        + (winner_index % kernel_w) * dilation_w
                    )
                    prior = input_gradient[batch_index, channel, input_y, input_x]
                    update = gradient[batch_index, channel, output_y, output_x]
                    input_gradient[batch_index, channel, input_y, input_x] = saturate_signed(
                        np.asarray(prior + update, dtype=np.int64), 32
                    )
    return input_gradient, winners


def execute_maxpool2d_backward(
    driver: TrainingCommandExecutor,
    values,
    output_gradients,
    *,
    kernel_size=3,
    stride=2,
    padding=1,
    dilation=1,
    timeout_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run max-pool winner selection and gradient scatter through vector RTL."""

    source = np.asarray(values, dtype=np.int64)
    gradient = np.asarray(output_gradients, dtype=np.int64)
    if source.ndim != 4 or gradient.ndim != 4:
        raise ValueError("max-pool values and gradients must be NCHW tensors")
    kernel_h, kernel_w = _pair("kernel_size", kernel_size)
    stride_h, stride_w = _pair("stride", stride)
    dilation_h, dilation_w = _pair("dilation", dilation)
    if isinstance(padding, int):
        pad_h, pad_w = int(padding), int(padding)
    else:
        pad_h, pad_w = tuple(map(int, padding))
    batch, channels, height, width = source.shape
    output_height = (
        height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_width = (
        width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if gradient.shape != (batch, channels, output_height, output_width):
        raise ValueError("max-pool output gradient shape does not match its geometry")
    windows = tuple(
        (batch_index, channel, output_y, output_x)
        for batch_index in range(batch)
        for channel in range(channels)
        for output_y in range(output_height)
        for output_x in range(output_width)
    )
    input_gradient = np.zeros(source.shape, dtype=np.int64)
    winners = np.zeros(gradient.shape, dtype=np.int64)

    for window_start in range(0, len(windows), 8):
        active_windows = windows[window_start:window_start + 8]
        active = len(active_windows)
        first_indices: list[int] = []
        current_values: list[int] = []
        for batch_index, channel, output_y, output_x in active_windows:
            found = False
            for kernel_y in range(kernel_h):
                input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                for kernel_x in range(kernel_w):
                    input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                    if 0 <= input_y < height and 0 <= input_x < width:
                        first_indices.append(kernel_y * kernel_w + kernel_x)
                        current_values.append(int(source[batch_index, channel, input_y, input_x]))
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError("a max-pool window contains no source elements")
        current = np.asarray(current_values + [0] * (8 - active), dtype=np.int64)
        winner = np.asarray(first_indices + [0] * (8 - active), dtype=np.int64)
        for candidate_index in range(kernel_h * kernel_w):
            candidates = current.copy()
            for lane, (batch_index, channel, output_y, output_x) in enumerate(active_windows):
                kernel_y, kernel_x = divmod(candidate_index, kernel_w)
                input_y = output_y * stride_h - pad_h + kernel_y * dilation_h
                input_x = output_x * stride_w - pad_w + kernel_x * dilation_w
                if (candidate_index > first_indices[lane]
                        and 0 <= input_y < height and 0 <= input_x < width):
                    candidates[lane] = source[batch_index, channel, input_y, input_x]
            keep_mask = _execute_vector8(
                driver,
                VectorOperation.COMPARE_GE,
                current,
                candidates,
                timeout_seconds=timeout_seconds,
            )
            current = _execute_vector8(
                driver,
                VectorOperation.MAXIMUM,
                current,
                candidates,
                timeout_seconds=timeout_seconds,
            )
            kept_winner = _execute_vector8(
                driver,
                VectorOperation.SELECT_NONZERO,
                keep_mask,
                winner,
                timeout_seconds=timeout_seconds,
            )
            replacement = _execute_vector8(
                driver,
                VectorOperation.SELECT_ZERO,
                keep_mask,
                (candidate_index,) * 8,
                timeout_seconds=timeout_seconds,
            )
            winner = _execute_vector8(
                driver,
                VectorOperation.ADD_SAT,
                kept_winner,
                replacement,
                timeout_seconds=timeout_seconds,
            )

        updates: list[tuple[tuple[int, int, int, int], int]] = []
        for lane, window in enumerate(active_windows):
            batch_index, channel, output_y, output_x = window
            winner_index = int(winner[lane])
            winners[window] = winner_index
            input_y = (
                output_y * stride_h - pad_h
                + (winner_index // kernel_w) * dilation_h
            )
            input_x = (
                output_x * stride_w - pad_w
                + (winner_index % kernel_w) * dilation_w
            )
            updates.append(
                ((batch_index, channel, input_y, input_x), int(gradient[window]))
            )

        # Each vector command updates distinct addresses. Repeated winners are
        # deferred to a later round, preserving output-window order per address.
        pending = list(updates)
        while pending:
            selected: list[tuple[tuple[int, int, int, int], int]] = []
            deferred: list[tuple[tuple[int, int, int, int], int]] = []
            seen: set[tuple[int, int, int, int]] = set()
            for update in pending:
                if update[0] in seen or len(selected) == 8:
                    deferred.append(update)
                else:
                    selected.append(update)
                    seen.add(update[0])
            left = [int(input_gradient[address]) for address, _ in selected]
            right = [value for _, value in selected]
            left.extend([0] * (8 - len(selected)))
            right.extend([0] * (8 - len(selected)))
            summed = _execute_vector8(
                driver,
                VectorOperation.ADD_SAT,
                left,
                right,
                timeout_seconds=timeout_seconds,
            )
            for lane, (address, _) in enumerate(selected):
                input_gradient[address] = summed[lane]
            pending = deferred
    return input_gradient, winners


def _shape_elements(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def linear_backward_partial_tiles(
    rows: int,
    inputs: int,
    outputs: int,
    *,
    repetitions: int = 1,
) -> int:
    """Count dX plus dW partial commands without materializing operands."""

    if min(rows, inputs, outputs, repetitions) <= 0:
        raise ValueError("linear backward dimensions and repetitions must be positive")
    return (
        2
        * ceil(rows / TILE)
        * ceil(inputs / TILE)
        * ceil(outputs / TILE)
        * repetitions
    )


def build_spikingformer_training_coverage(
    execution_manifest_path: str | Path,
    parameter_manifest_path: str | Path,
) -> dict[str, object]:
    """Prove hardware backward/update coverage for the exported deployed graph."""

    execution_path = Path(execution_manifest_path).expanduser().resolve()
    parameter_path = Path(parameter_manifest_path).expanduser().resolve()
    execution = json.loads(execution_path.read_text(encoding="ascii"))
    parameters = json.loads(parameter_path.read_text(encoding="ascii"))
    operators = {item["name"]: item for item in execution["operators"]}
    projection_entries = {item["name"]: item for item in parameters["projections"]}
    learned_names = tuple(projection_entries) + ("head",)
    missing_operators = sorted(set(learned_names) - set(operators))
    if missing_operators:
        raise ValueError(f"learned parameter groups lack graph operators: {missing_operators}")

    learned_coverage: list[dict[str, object]] = []
    learned_parameter_elements = 0
    learned_backward_tiles = 0
    learned_backward_multiplications = 0
    for name in learned_names:
        operator = operators[name]
        matrix = operator["matrix"]
        rows = int(matrix["rows"])
        inputs = int(matrix["reduction"])
        outputs = int(matrix["columns"])
        repetitions = int(matrix.get("repetitions", 1))
        augmented_inputs = inputs + 1
        partial_tiles = linear_backward_partial_tiles(
            rows,
            augmented_inputs,
            outputs,
            repetitions=repetitions,
        )
        multiplications = 2 * rows * augmented_inputs * outputs * repetitions
        if name == "head":
            entry = parameters["head"]
            weight_elements = _shape_elements(entry["hardware_matrix_shape"])
            bias_elements = int(entry["hardware_matrix_shape"][1])
        else:
            entry = projection_entries[name]
            weight_elements = _shape_elements(entry["hardware_matrix_shape"])
            bias_elements = int(entry["hardware_matrix_shape"][1])
        parameter_elements = weight_elements + bias_elements
        learned_parameter_elements += parameter_elements
        learned_backward_tiles += partial_tiles
        learned_backward_multiplications += multiplications
        learned_coverage.append({
            "name": name,
            "rows": rows,
            "inputs": inputs,
            "outputs": outputs,
            "repetitions": repetitions,
            "weight_elements": weight_elements,
            "bias_elements": bias_elements,
            "trainable_deployed_elements": parameter_elements,
            "backward_partial_tiles": partial_tiles,
            "backward_multiplications": multiplications,
            "gradient_method": "47-bit raw accumulation for dX/dW; homogeneous bias row",
            "update_method": "eight-lane signed fixed-point SGDW in programmable logic",
        })

    dynamic_names = tuple(
        name for name in operators if name.endswith(".attn.qk") or name.endswith(".attn.av")
    )
    dynamic_coverage: list[dict[str, object]] = []
    dynamic_tiles = 0
    dynamic_multiplications = 0
    for name in dynamic_names:
        matrix = operators[name]["matrix"]
        rows = int(matrix["rows"])
        inputs = int(matrix["reduction"])
        outputs = int(matrix["columns"])
        repetitions = int(matrix.get("repetitions", 1))
        partial_tiles = linear_backward_partial_tiles(
            rows, inputs, outputs, repetitions=repetitions
        )
        multiplications = 2 * rows * inputs * outputs * repetitions
        dynamic_tiles += partial_tiles
        dynamic_multiplications += multiplications
        dynamic_coverage.append({
            "name": name,
            "rows": rows,
            "inputs": inputs,
            "outputs": outputs,
            "repetitions": repetitions,
            "backward_partial_tiles": partial_tiles,
            "backward_multiplications": multiplications,
            "gradient_method": "two signed raw-accumulator matrix passes",
        })

    operator_coverage: list[dict[str, str]] = []
    uncovered: list[str] = []
    for name, operator in operators.items():
        source = str(operator["source_operation"])
        if name in learned_names:
            method = "affine dX/dW/dB plus SGDW"
            if "MaxPool2d" in source:
                method += "; first-winner max-pool backward and fabric scatter-add"
            if "LIF" in source:
                method += "; traced sigmoid-surrogate hard-LIF BPTT"
        elif name in dynamic_names:
            method = "dynamic matrix dX/dW; 0.125 attention gradient uses Q15 multiply"
        elif name.endswith(".residuals"):
            method = "identity gradient fan-out; converging branches use saturating vector add"
        elif name == "global_average_pool":
            method = "gradient broadcast with exact power-of-two division by 256"
        else:
            uncovered.append(name)
            method = "UNMAPPED"
        operator_coverage.append({"name": name, "backward_method": method})
    if uncovered:
        raise ValueError(f"execution operators lack backward methods: {uncovered}")

    depth = int(execution["model"]["transformer_blocks"])
    lif_nodes = 4 + 7 * depth
    optimizer_commands = ceil(learned_parameter_elements / 8)
    return {
        "format": "neuraedge-spikingformer-training-coverage-v1",
        "execution_manifest": str(execution_path),
        "parameter_manifest": str(parameter_path),
        "training_semantics": {
            "target_graph": "complete exported integer inference graph",
            "quantizer_gradient": "straight-through estimator with fixed exported scales",
            "lif_gradient": "SpikingJelly sigmoid surrogate alpha=4, hard reset, detach_reset=True",
            "temporal_method": "four-step BPTT in reverse timestep order through cached membrane traces",
            "optimizer": (
                "signed fixed-point SGDW with momentum; weight decay is applied "
                "to the update direction and excluded from retained velocity"
            ),
            "parameter_residency": "PYNQ DDR master tensors; arithmetic and updates in programmable logic",
        },
        "summary": {
            "execution_operator_count": len(operators),
            "covered_execution_operator_count": len(operator_coverage),
            "learned_affine_group_count": len(learned_names),
            "backbone_projection_count": len(projection_entries),
            "dynamic_attention_matmul_count": len(dynamic_names),
            "lif_node_count": lif_nodes,
            "maxpool_backward_count": 2,
            "residual_gradient_fanout_count": 2 * depth,
            "trainable_deployed_parameter_elements": learned_parameter_elements,
            "sgdw_update_commands": optimizer_commands,
            "learned_backward_partial_tiles": learned_backward_tiles,
            "dynamic_backward_partial_tiles": dynamic_tiles,
            "total_backward_partial_tiles": learned_backward_tiles + dynamic_tiles,
            "learned_backward_multiplications": learned_backward_multiplications,
            "dynamic_backward_multiplications": dynamic_multiplications,
            "full_deployed_backbone_covered": True,
            "uncovered_operator_count": 0,
        },
        "learned_affine_coverage": learned_coverage,
        "dynamic_attention_coverage": dynamic_coverage,
        "operator_coverage": operator_coverage,
    }
