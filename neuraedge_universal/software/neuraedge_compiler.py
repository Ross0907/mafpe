"""Tiled uint8 GEMM compiler for the NeuraEdge tensor engine.

The compiler turns a mathematical ``(A - za) @ (B - zb) + bias`` operation
into the exact load/compute transactions accepted by the RTL.  It deliberately
keeps compilation separate from execution: plans can be inspected, tested,
serialized, or sent to a PYNQ board through :class:`NeuraEdgeDriver`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from neuraedge_golden import Dataflow, choose_dataflow
from neuraedge_protocol import (
    Command128,
    Opcode,
    PackedStreamBeats,
    ProgramRecord,
    StreamBeat,
    tensor_command,
    unpack_tensor_result,
)


ROW_TILE = 8
COLUMN_TILE = 16
MAX_REDUCTION = 2048


def _pack_compiler_u8_rows(values: np.ndarray, row_bytes: int) -> PackedStreamBeats:
    """Pack validated uint8 rows into 64-bit stream words without Python lane loops."""

    array = np.asarray(values)
    if array.dtype != np.uint8 or array.ndim < 2 or array.shape[-1] != row_bytes:
        raise ValueError(f"compiler stream tensor must be uint8 with final dimension {row_bytes}")
    if row_bytes % 8:
        raise ValueError("compiler stream row width must be a multiple of eight")
    lanes = np.ascontiguousarray(array).reshape(-1, 8).astype(np.uint64)
    shifts = (np.arange(8, dtype=np.uint64) * np.uint64(8)).reshape(1, 8)
    words = np.bitwise_or.reduce(lanes << shifts, axis=1).astype("<u8", copy=False)
    return PackedStreamBeats.from_little_endian_bytes(words.tobytes(order="C"))


class CommandExecutor(Protocol):
    """Smallest driver surface required to run a compiled program."""

    def execute(
        self,
        command: Command128,
        input_beats: Sequence[StreamBeat] = (),
        *,
        output_beats: int = 0,
        timeout_seconds: float = 5.0,
    ) -> tuple[int, ...]: ...


class ProgramExecutor(CommandExecutor, Protocol):
    def execute_program(
        self,
        records: Sequence[ProgramRecord],
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class TensorInstruction:
    """One atomic command, its input stream, and expected output length."""

    command: Command128
    input_beats: Sequence[StreamBeat] = ()
    output_beats: int = 0
    description: str = ""


@dataclass(frozen=True)
class ResultBinding:
    """Maps local row/column result tags back to the destination matrix."""

    instruction_index: int
    row_start: int
    column_start: int
    rows: int
    columns: int
    reduction_start: int
    reduction_count: int


@dataclass(frozen=True)
class CompiledGemm:
    """Executable, inspectable transaction program for one uint8 GEMM."""

    shape: tuple[int, int, int]
    activation_zero_point: int
    weight_zero_point: int
    instructions: tuple[TensorInstruction, ...]
    result_bindings: tuple[ResultBinding, ...]
    bias: tuple[int, ...] | None
    estimated_input_bytes: int
    mathematical_macs: int
    estimated_spike_adds: int = 0

    def execute(self, driver: CommandExecutor, *, timeout_seconds: float = 5.0) -> np.ndarray:
        """Run every transaction and reconstruct the signed int64 matrix."""
        rows, _, columns = self.shape
        output = np.zeros((rows, columns), dtype=np.int64)
        bindings = {binding.instruction_index: binding for binding in self.result_bindings}

        for instruction_index, instruction in enumerate(self.instructions):
            words = driver.execute(
                instruction.command,
                instruction.input_beats,
                output_beats=instruction.output_beats,
                timeout_seconds=timeout_seconds,
            )
            if len(words) != instruction.output_beats:
                raise RuntimeError(
                    f"instruction {instruction_index} returned {len(words)} beats; "
                    f"expected {instruction.output_beats}"
                )
            binding = bindings.get(instruction_index)
            if binding is None:
                continue

            _accumulate_tensor_tile(output, binding, words)

        if self.bias is not None:
            output += np.asarray(self.bias, dtype=np.int64).reshape(1, columns)
        return output

    def execute_batched(
        self,
        driver: ProgramExecutor,
        *,
        timeout_seconds: float = 30.0,
    ) -> np.ndarray:
        """Run every tile through one descriptor-program DMA transaction."""

        rows, _, columns = self.shape
        records = tuple(ProgramRecord(
            instruction.command,
            instruction.input_beats,
            instruction.output_beats,
        ) for instruction in self.instructions)
        flat_words = driver.execute_program(records, timeout_seconds=timeout_seconds)
        expected_words = sum(instruction.output_beats for instruction in self.instructions)
        if len(flat_words) != expected_words:
            raise RuntimeError(
                f"descriptor program returned {len(flat_words)} beats; expected {expected_words}")
        output = np.zeros((rows, columns), dtype=np.int64)
        bindings = {binding.instruction_index: binding for binding in self.result_bindings}
        word_offset = 0
        for instruction_index, instruction in enumerate(self.instructions):
            word_end = word_offset + instruction.output_beats
            words = flat_words[word_offset:word_end]
            binding = bindings.get(instruction_index)
            if binding is not None:
                _accumulate_tensor_tile(output, binding, words)
            word_offset = word_end
        if self.bias is not None:
            output += np.asarray(self.bias, dtype=np.int64).reshape(1, columns)
        return output


def _accumulate_tensor_tile(
    output: np.ndarray,
    binding: ResultBinding,
    words: Sequence[int],
) -> None:
    seen: set[tuple[int, int]] = set()
    for beat_index, word in enumerate(words):
        value, local_row, local_column, last = unpack_tensor_result(word)
        if local_row >= binding.rows or local_column >= binding.columns:
            raise RuntimeError(
                f"out-of-range tensor tag ({local_row},{local_column}) "
                f"for {binding.rows}x{binding.columns} result tile")
        tag = (local_row, local_column)
        if tag in seen:
            raise RuntimeError(f"duplicate tensor result tag {tag}")
        if last != (beat_index == len(words) - 1):
            raise RuntimeError("tensor TLAST metadata does not match the final result beat")
        seen.add(tag)
        output[binding.row_start + local_row,
               binding.column_start + local_column] += value
    if len(seen) != binding.rows * binding.columns:
        raise RuntimeError("tensor result tile did not contain every row/column tag")


@dataclass(frozen=True)
class ConvGroupProgram:
    output_channel_start: int
    output_channel_count: int
    gemm: CompiledGemm


@dataclass(frozen=True)
class CompiledConv2d:
    """One or more GEMM programs plus the exact NCHW reconstruction map."""

    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    group_programs: tuple[ConvGroupProgram, ...]

    def execute(self, driver: CommandExecutor, *, timeout_seconds: float = 5.0) -> np.ndarray:
        batch, output_channels, output_height, output_width = self.output_shape
        spatial_rows = batch * output_height * output_width
        flat = np.zeros((spatial_rows, output_channels), dtype=np.int64)
        for group in self.group_programs:
            values = group.gemm.execute(driver, timeout_seconds=timeout_seconds)
            start = group.output_channel_start
            flat[:, start:start + group.output_channel_count] = values
        return flat.reshape(batch, output_height, output_width, output_channels).transpose(0, 3, 1, 2)

    def execute_batched(
        self,
        driver: ProgramExecutor,
        *,
        timeout_seconds: float = 30.0,
    ) -> np.ndarray:
        batch, output_channels, output_height, output_width = self.output_shape
        spatial_rows = batch * output_height * output_width
        flat = np.zeros((spatial_rows, output_channels), dtype=np.int64)
        for group in self.group_programs:
            values = group.gemm.execute_batched(
                driver, timeout_seconds=timeout_seconds)
            start = group.output_channel_start
            flat[:, start:start + group.output_channel_count] = values
        return flat.reshape(batch, output_height, output_width, output_channels).transpose(0, 3, 1, 2)

    @property
    def mathematical_macs(self) -> int:
        return sum(group.gemm.mathematical_macs for group in self.group_programs)

    @property
    def estimated_input_bytes(self) -> int:
        return sum(group.gemm.estimated_input_bytes for group in self.group_programs)

    @property
    def estimated_spike_adds(self) -> int:
        return sum(group.gemm.estimated_spike_adds for group in self.group_programs)


def _as_u8_matrix(name: str, value: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    as_i64 = array.astype(np.int64)
    if np.any(as_i64 < 0) or np.any(as_i64 > 255):
        raise ValueError(f"{name} entries must be in [0,255]")
    return as_i64.astype(np.uint8)


def _u8(name: str, value: int) -> int:
    integer = int(value)
    if integer < 0 or integer > 255:
        raise ValueError(f"{name} must be in [0,255]")
    return integer


def _pair(name: str, value: int | Sequence[int], *, allow_zero: bool = False) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or a two-element sequence")
        pair = (int(value[0]), int(value[1]))
    else:
        pair = (int(value), int(value))
    minimum = 0 if allow_zero else 1
    if pair[0] < minimum or pair[1] < minimum:
        raise ValueError(f"{name} values must be at least {minimum}")
    return pair


def _im2col_nchw(
    activations: np.ndarray,
    kernel: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    pad_value: int,
) -> tuple[np.ndarray, int, int]:
    batch, channels, height, width = activations.shape
    kernel_height, kernel_width = kernel
    effective_height = dilation[0] * (kernel_height - 1) + 1
    effective_width = dilation[1] * (kernel_width - 1) + 1
    numerator_height = height + 2 * padding[0] - effective_height
    numerator_width = width + 2 * padding[1] - effective_width
    if numerator_height < 0 or numerator_width < 0:
        raise ValueError("effective convolution kernel is larger than the padded input")
    output_height = numerator_height // stride[0] + 1
    output_width = numerator_width // stride[1] + 1
    padded = np.full(
        (batch, channels, height + 2 * padding[0], width + 2 * padding[1]),
        pad_value,
        dtype=activations.dtype,
    )
    padded[:, :, padding[0]:padding[0] + height, padding[1]:padding[1] + width] = activations
    columns = np.empty(
        (batch * output_height * output_width, channels * kernel_height * kernel_width),
        dtype=activations.dtype,
    )
    row = 0
    for batch_index in range(batch):
        for output_y in range(output_height):
            input_y = output_y * stride[0]
            for output_x in range(output_width):
                input_x = output_x * stride[1]
                column = 0
                for channel in range(channels):
                    for kernel_y in range(kernel_height):
                        y = input_y + kernel_y * dilation[0]
                        for kernel_x in range(kernel_width):
                            x = input_x + kernel_x * dilation[1]
                            columns[row, column] = padded[batch_index, channel, y, x]
                            column += 1
                row += 1
    return columns, output_height, output_width


def compile_u8_gemm(
    activations: np.ndarray | Sequence[Sequence[int]],
    weights: np.ndarray | Sequence[Sequence[int]],
    *,
    activation_zero_point: int = 0,
    weight_zero_point: int = 0,
    bias: Sequence[int] | np.ndarray | None = None,
    dataflow: Dataflow | None = None,
) -> CompiledGemm:
    """Compile an exact asymmetric-uint8 matrix multiplication.

    Reduction dimensions above 2,048 are split into independently corrected
    chunks and accumulated by the runtime.  A single-row matrix uses Dot8 mode;
    all other shapes use the 8x16 outer-product mode.
    """
    a = _as_u8_matrix("activations", activations)
    b = _as_u8_matrix("weights", weights)
    rows, reduction = a.shape
    weight_reduction, columns = b.shape
    if reduction != weight_reduction:
        raise ValueError("activation columns must equal weight rows")
    za = _u8("activation_zero_point", activation_zero_point)
    zb = _u8("weight_zero_point", weight_zero_point)

    bias_tuple: tuple[int, ...] | None = None
    if bias is not None:
        bias_array = np.asarray(bias)
        if bias_array.ndim != 1 or len(bias_array) != columns:
            raise ValueError("bias must be a one-dimensional vector with one entry per output column")
        if not np.issubdtype(bias_array.dtype, np.integer):
            raise ValueError("bias must contain integers")
        bias_tuple = tuple(int(value) for value in bias_array)

    instructions: list[TensorInstruction] = []
    bindings: list[ResultBinding] = []
    estimated_bytes = 0

    def append_load(command: Command128, beats: Sequence[StreamBeat], description: str) -> None:
        nonlocal estimated_bytes
        instructions.append(TensorInstruction(command, beats, 0, description))
        estimated_bytes += 8 * len(beats)

    def append_compute(
        command: Command128,
        row_start: int,
        column_start: int,
        tile_rows: int,
        tile_columns: int,
        reduction_start: int,
        reduction_count: int,
    ) -> None:
        instruction_index = len(instructions)
        instructions.append(TensorInstruction(
            command, (), tile_rows * tile_columns,
            f"compute C[{row_start}:{row_start + tile_rows},"
            f"{column_start}:{column_start + tile_columns}] "
            f"from K[{reduction_start}:{reduction_start + reduction_count}]",
        ))
        bindings.append(ResultBinding(
            instruction_index, row_start, column_start, tile_rows, tile_columns,
            reduction_start, reduction_count,
        ))

    for reduction_start in range(0, reduction, MAX_REDUCTION):
        reduction_count = min(MAX_REDUCTION, reduction - reduction_start)
        selected = dataflow or choose_dataflow(rows, columns, reduction_count).mode
        if selected not in {Dataflow.INPUT_BROADCAST, Dataflow.WEIGHT_BROADCAST}:
            raise ValueError(f"unsupported dataflow {selected}")
        use_dot8 = rows == 1
        issue_count = (reduction_count + 7) // 8 if use_dot8 else reduction_count
        load_weight_opcode = Opcode.LOAD_WEIGHTS_DOT8 if use_dot8 else Opcode.LOAD_WEIGHTS_OUTER
        compute_opcode = Opcode.COMPUTE_DENSE_DOT8 if use_dot8 else Opcode.COMPUTE_DENSE_OUTER

        def activation_load(row_start: int) -> tuple[Command128, PackedStreamBeats]:
            tile_rows = min(ROW_TILE, rows - row_start)
            if use_dot8:
                padded = np.full(issue_count * 8, za, dtype=np.uint8)
                padded[:reduction_count] = a[row_start, reduction_start:reduction_start + reduction_count]
                vectors = padded.reshape(issue_count, 8)
            else:
                vectors = np.full((reduction_count, ROW_TILE), za, dtype=np.uint8)
                vectors[:, :tile_rows] = a[
                    row_start:row_start + tile_rows,
                    reduction_start:reduction_start + reduction_count,
                ].T
            command = tensor_command(
                Opcode.LOAD_ACTIVATIONS,
                issue_count=issue_count,
                activation_zero_point=za,
            )
            return command, _pack_compiler_u8_rows(vectors, 8)

        def weight_load(column_start: int) -> tuple[Command128, PackedStreamBeats]:
            tile_columns = min(COLUMN_TILE, columns - column_start)
            if use_dot8:
                tensor = np.full((issue_count, 8, COLUMN_TILE), zb, dtype=np.uint8)
                flat = tensor.reshape(issue_count * 8, COLUMN_TILE)
                flat[:reduction_count, :tile_columns] = b[
                    reduction_start:reduction_start + reduction_count,
                    column_start:column_start + tile_columns,
                ]
                beats = _pack_compiler_u8_rows(tensor, 16)
            else:
                matrix = np.full((reduction_count, COLUMN_TILE), zb, dtype=np.uint8)
                matrix[:, :tile_columns] = b[
                    reduction_start:reduction_start + reduction_count,
                    column_start:column_start + tile_columns,
                ]
                beats = _pack_compiler_u8_rows(matrix, 16)
            command = tensor_command(load_weight_opcode, issue_count=issue_count)
            return command, beats

        def compute(row_start: int, column_start: int) -> None:
            tile_rows = min(ROW_TILE, rows - row_start)
            tile_columns = min(COLUMN_TILE, columns - column_start)
            command = tensor_command(
                compute_opcode,
                rows=tile_rows,
                columns=tile_columns,
                issue_count=issue_count,
                activation_zero_point=za,
                weight_zero_point=zb,
            )
            append_compute(
                command, row_start, column_start, tile_rows, tile_columns,
                reduction_start, reduction_count,
            )

        if selected == Dataflow.INPUT_BROADCAST:
            for row_start in range(0, rows, ROW_TILE):
                command, beats = activation_load(row_start)
                append_load(command, beats, f"load activation tile at row {row_start}")
                for column_start in range(0, columns, COLUMN_TILE):
                    command, beats = weight_load(column_start)
                    append_load(command, beats, f"load weight tile at column {column_start}")
                    compute(row_start, column_start)
        else:
            for column_start in range(0, columns, COLUMN_TILE):
                command, beats = weight_load(column_start)
                append_load(command, beats, f"load weight tile at column {column_start}")
                for row_start in range(0, rows, ROW_TILE):
                    command, beats = activation_load(row_start)
                    append_load(command, beats, f"load activation tile at row {row_start}")
                    compute(row_start, column_start)

    return CompiledGemm(
        (rows, reduction, columns), za, zb, tuple(instructions), tuple(bindings),
        bias_tuple, estimated_bytes, rows * reduction * columns,
    )


def compile_spike_gemm(
    spikes: np.ndarray | Sequence[Sequence[int]],
    weights: np.ndarray | Sequence[Sequence[int]],
    *,
    ternary: bool = False,
    bias: Sequence[int] | np.ndarray | None = None,
    dataflow: Dataflow | None = None,
) -> CompiledGemm:
    """Compile binary or ternary spikes times signed INT8 weights.

    The RTL performs one signed addition for each non-zero spike/output-column
    pair and performs no multiplication.  Ternary ``-1`` spikes select
    subtraction.  Zero-spike K positions are compacted by the activation-load
    pipeline and do not reach the tensor tile.
    """
    spike_array = np.asarray(spikes)
    weight_array = np.asarray(weights)
    if spike_array.ndim != 2 or weight_array.ndim != 2:
        raise ValueError("spikes and weights must be two-dimensional matrices")
    if spike_array.size == 0 or weight_array.size == 0:
        raise ValueError("spikes and weights must not be empty")
    if not np.issubdtype(spike_array.dtype, np.integer) or not np.issubdtype(weight_array.dtype, np.integer):
        raise ValueError("spikes and weights must contain integers")
    spike_i64 = spike_array.astype(np.int64)
    weight_i64 = weight_array.astype(np.int64)
    allowed = np.isin(spike_i64, (-1, 0, 1) if ternary else (0, 1))
    if not np.all(allowed):
        expected = "{-1,0,1}" if ternary else "{0,1}"
        raise ValueError(f"spikes must be in {expected}")
    if np.any(weight_i64 < -128) or np.any(weight_i64 > 127):
        raise ValueError("sparse-mode weights must be signed INT8 values")

    rows, reduction = spike_i64.shape
    weight_reduction, columns = weight_i64.shape
    if reduction != weight_reduction:
        raise ValueError("spike columns must equal weight rows")
    bias_tuple: tuple[int, ...] | None = None
    if bias is not None:
        bias_array = np.asarray(bias)
        if bias_array.ndim != 1 or len(bias_array) != columns:
            raise ValueError("bias must have one entry per output column")
        if not np.issubdtype(bias_array.dtype, np.integer):
            raise ValueError("bias must contain integers")
        bias_tuple = tuple(int(value) for value in bias_array)

    encoded_spikes = (spike_i64 & 0xFF).astype(np.uint8)
    encoded_weights = (weight_i64 & 0xFF).astype(np.uint8)
    instructions: list[TensorInstruction] = []
    bindings: list[ResultBinding] = []
    estimated_bytes = 0
    compute_opcode = Opcode.COMPUTE_TERNARY if ternary else Opcode.COMPUTE_BINARY

    def append_load(command: Command128, beats: Sequence[StreamBeat], description: str) -> None:
        nonlocal estimated_bytes
        instructions.append(TensorInstruction(command, beats, 0, description))
        estimated_bytes += 8 * len(beats)

    for reduction_start in range(0, reduction, MAX_REDUCTION):
        reduction_count = min(MAX_REDUCTION, reduction - reduction_start)
        selected = dataflow or choose_dataflow(rows, columns, reduction_count).mode
        if selected not in {Dataflow.INPUT_BROADCAST, Dataflow.WEIGHT_BROADCAST}:
            raise ValueError(f"unsupported dataflow {selected}")

        def load_activations(row_start: int) -> None:
            tile_rows = min(ROW_TILE, rows - row_start)
            vectors = np.zeros((reduction_count, ROW_TILE), dtype=np.uint8)
            vectors[:, :tile_rows] = encoded_spikes[
                row_start:row_start + tile_rows,
                reduction_start:reduction_start + reduction_count,
            ].T
            append_load(
                tensor_command(Opcode.LOAD_ACTIVATIONS, issue_count=reduction_count),
                _pack_compiler_u8_rows(vectors, 8),
                f"load {'ternary' if ternary else 'binary'} spike tile at row {row_start}",
            )

        def load_weights(column_start: int) -> None:
            tile_columns = min(COLUMN_TILE, columns - column_start)
            matrix = np.zeros((reduction_count, COLUMN_TILE), dtype=np.uint8)
            matrix[:, :tile_columns] = encoded_weights[
                reduction_start:reduction_start + reduction_count,
                column_start:column_start + tile_columns,
            ]
            append_load(
                tensor_command(Opcode.LOAD_WEIGHTS_OUTER, issue_count=reduction_count),
                _pack_compiler_u8_rows(matrix, 16),
                f"load signed weight tile at column {column_start}",
            )

        def append_compute(row_start: int, column_start: int) -> None:
            tile_rows = min(ROW_TILE, rows - row_start)
            tile_columns = min(COLUMN_TILE, columns - column_start)
            instruction_index = len(instructions)
            instructions.append(TensorInstruction(
                tensor_command(
                    compute_opcode,
                    rows=tile_rows,
                    columns=tile_columns,
                    issue_count=reduction_count,
                ),
                (),
                tile_rows * tile_columns,
                f"compute sparse C tile at ({row_start},{column_start})",
            ))
            bindings.append(ResultBinding(
                instruction_index, row_start, column_start, tile_rows, tile_columns,
                reduction_start, reduction_count,
            ))

        if selected == Dataflow.INPUT_BROADCAST:
            for row_start in range(0, rows, ROW_TILE):
                load_activations(row_start)
                for column_start in range(0, columns, COLUMN_TILE):
                    load_weights(column_start)
                    append_compute(row_start, column_start)
        else:
            for column_start in range(0, columns, COLUMN_TILE):
                load_weights(column_start)
                for row_start in range(0, rows, ROW_TILE):
                    load_activations(row_start)
                    append_compute(row_start, column_start)

    spike_adds = int(np.count_nonzero(spike_i64)) * columns
    return CompiledGemm(
        (rows, reduction, columns), 0, 0, tuple(instructions), tuple(bindings),
        bias_tuple, estimated_bytes, rows * reduction * columns, spike_adds,
    )


def compile_u8_conv2d(
    activations: np.ndarray | Sequence,
    weights: np.ndarray | Sequence,
    *,
    activation_zero_point: int = 0,
    weight_zero_point: int = 0,
    bias: Sequence[int] | np.ndarray | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    dataflow: Dataflow | None = None,
) -> CompiledConv2d:
    """Lower an NCHW/OIHW asymmetric-uint8 convolution to tensor commands."""
    a = np.asarray(activations)
    w = np.asarray(weights)
    if a.ndim != 4 or w.ndim != 4:
        raise ValueError("convolution activations and weights must be NCHW and OIHW tensors")
    if not np.issubdtype(a.dtype, np.integer) or not np.issubdtype(w.dtype, np.integer):
        raise ValueError("convolution activations and weights must contain integers")
    a_i64 = a.astype(np.int64)
    w_i64 = w.astype(np.int64)
    if np.any(a_i64 < 0) or np.any(a_i64 > 255):
        raise ValueError("convolution activations must be in [0,255]")
    if np.any(w_i64 < 0) or np.any(w_i64 > 255):
        raise ValueError("convolution weights must be in [0,255]")
    a_u8 = a_i64.astype(np.uint8)
    w_u8 = w_i64.astype(np.uint8)
    za = _u8("activation_zero_point", activation_zero_point)
    zb = _u8("weight_zero_point", weight_zero_point)
    stride_pair = _pair("stride", stride)
    padding_pair = _pair("padding", padding, allow_zero=True)
    dilation_pair = _pair("dilation", dilation)

    batch, input_channels, height, width = a_u8.shape
    output_channels, weight_channels, kernel_height, kernel_width = w_u8.shape
    group_count = int(groups)
    if group_count <= 0 or input_channels % group_count or output_channels % group_count:
        raise ValueError("groups must divide both input and output channels")
    channels_per_group = input_channels // group_count
    outputs_per_group = output_channels // group_count
    if weight_channels != channels_per_group:
        raise ValueError("OIHW weight input channels must equal input_channels/groups")
    if kernel_height == 0 or kernel_width == 0:
        raise ValueError("convolution kernel dimensions must be non-zero")
    if bias is None:
        bias_values = None
    else:
        bias_array = np.asarray(bias)
        if bias_array.ndim != 1 or len(bias_array) != output_channels:
            raise ValueError("convolution bias must have one entry per output channel")
        if not np.issubdtype(bias_array.dtype, np.integer):
            raise ValueError("convolution bias must contain integers")
        bias_values = bias_array.astype(np.int64)

    programs: list[ConvGroupProgram] = []
    output_height = output_width = 0
    for group in range(group_count):
        input_start = group * channels_per_group
        output_start = group * outputs_per_group
        columns, output_height, output_width = _im2col_nchw(
            a_u8[:, input_start:input_start + channels_per_group],
            (kernel_height, kernel_width), stride_pair, padding_pair,
            dilation_pair, za,
        )
        weight_matrix = w_u8[
            output_start:output_start + outputs_per_group
        ].reshape(outputs_per_group, -1).T
        group_bias = None if bias_values is None else bias_values[
            output_start:output_start + outputs_per_group
        ]
        programs.append(ConvGroupProgram(
            output_start,
            outputs_per_group,
            compile_u8_gemm(
                columns,
                weight_matrix,
                activation_zero_point=za,
                weight_zero_point=zb,
                bias=group_bias,
                dataflow=dataflow,
            ),
        ))
    return CompiledConv2d(
        (batch, input_channels, height, width),
        (batch, output_channels, output_height, output_width),
        (kernel_height, kernel_width), stride_pair, padding_pair, dilation_pair,
        group_count, tuple(programs),
    )


def compile_spike_conv2d(
    spikes: np.ndarray | Sequence,
    weights: np.ndarray | Sequence,
    *,
    ternary: bool = False,
    bias: Sequence[int] | np.ndarray | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
    dataflow: Dataflow | None = None,
) -> CompiledConv2d:
    """Lower grouped binary/ternary spike convolution to addition-only GEMMs."""
    spike_array = np.asarray(spikes)
    weight_array = np.asarray(weights)
    if spike_array.ndim != 4 or weight_array.ndim != 4:
        raise ValueError("spike convolution requires NCHW activations and OIHW weights")
    if not np.issubdtype(spike_array.dtype, np.integer) or not np.issubdtype(weight_array.dtype, np.integer):
        raise ValueError("spike convolution tensors must contain integers")
    spike_i64 = spike_array.astype(np.int64)
    weight_i64 = weight_array.astype(np.int64)
    if not np.all(np.isin(spike_i64, (-1, 0, 1) if ternary else (0, 1))):
        raise ValueError("spike values do not match the selected binary/ternary mode")
    if np.any(weight_i64 < -128) or np.any(weight_i64 > 127):
        raise ValueError("spike convolution weights must be signed INT8")
    stride_pair = _pair("stride", stride)
    padding_pair = _pair("padding", padding, allow_zero=True)
    dilation_pair = _pair("dilation", dilation)
    batch, input_channels, height, width = spike_i64.shape
    output_channels, weight_channels, kernel_height, kernel_width = weight_i64.shape
    group_count = int(groups)
    if group_count <= 0 or input_channels % group_count or output_channels % group_count:
        raise ValueError("groups must divide both input and output channels")
    channels_per_group = input_channels // group_count
    outputs_per_group = output_channels // group_count
    if weight_channels != channels_per_group:
        raise ValueError("OIHW weight input channels must equal input_channels/groups")
    if bias is None:
        bias_values = None
    else:
        bias_array = np.asarray(bias)
        if bias_array.ndim != 1 or len(bias_array) != output_channels:
            raise ValueError("spike convolution bias must have one entry per output channel")
        if not np.issubdtype(bias_array.dtype, np.integer):
            raise ValueError("spike convolution bias must contain integers")
        bias_values = bias_array.astype(np.int64)

    programs: list[ConvGroupProgram] = []
    output_height = output_width = 0
    for group in range(group_count):
        input_start = group * channels_per_group
        output_start = group * outputs_per_group
        columns, output_height, output_width = _im2col_nchw(
            spike_i64[:, input_start:input_start + channels_per_group],
            (kernel_height, kernel_width), stride_pair, padding_pair,
            dilation_pair, 0,
        )
        weight_matrix = weight_i64[
            output_start:output_start + outputs_per_group
        ].reshape(outputs_per_group, -1).T
        group_bias = None if bias_values is None else bias_values[
            output_start:output_start + outputs_per_group
        ]
        programs.append(ConvGroupProgram(
            output_start,
            outputs_per_group,
            compile_spike_gemm(
                columns,
                weight_matrix,
                ternary=ternary,
                bias=group_bias,
                dataflow=dataflow,
            ),
        ))
    return CompiledConv2d(
        (batch, input_channels, height, width),
        (batch, output_channels, output_height, output_width),
        (kernel_height, kernel_width), stride_pair, padding_pair, dilation_pair,
        group_count, tuple(programs),
    )
