"""Typed command and stream ABI for the NeuraEdge PYNQ accelerator.

The RTL command port is 128 bits wide and the data path is AXI4-Stream with
64-bit beats.  This module is deliberately independent of PYNQ and NumPy so it
can be used by fixture generators, desktop tests, and the board driver.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import IntEnum
from sys import byteorder
from typing import Iterable, Iterator, Sequence, overload


class Opcode(IntEnum):
    TRAIN_LINEAR_BACKWARD = 0x0
    LOAD_ACTIVATIONS = 0x1
    LOAD_WEIGHTS_OUTER = 0x2
    LOAD_WEIGHTS_DOT8 = 0x3
    COMPUTE_DENSE_OUTER = 0x4
    COMPUTE_DENSE_DOT8 = 0x5
    COMPUTE_BINARY = 0x6
    COMPUTE_TERNARY = 0x7
    ALPHA_XNOR = 0x8
    VECTOR_ALU = 0x9
    LAYERNORM = 0xA
    LOG_SOFTMAX = 0xB
    LIF_STEP = 0xC
    SGD_UPDATE = 0xD
    ACTIVATION_LUT = 0xE
    CLEAR_STATUS = 0xF


class LutSuboperation(IntEnum):
    CLEAR = 0
    LOAD = 1
    EVALUATE = 2


class LinearGradientMode(IntEnum):
    BOTH = 0
    INPUT = 1
    WEIGHT = 2


@dataclass(frozen=True)
class StreamBeat:
    data: int
    last: bool = False

    def __post_init__(self) -> None:
        _unsigned("stream data", self.data, 64)


class PackedStreamBeats(Sequence[StreamBeat]):
    """Compact 64-bit payload storage with the ordinary StreamBeat interface."""

    def __init__(self, words: Iterable[int] = ()) -> None:
        self.words = array("Q", (_unsigned("stream word", word, 64) for word in words))
        if self.words.itemsize != 8:
            raise RuntimeError("platform unsigned long long is not 64 bits")

    @classmethod
    def from_little_endian_bytes(cls, payload: bytes) -> "PackedStreamBeats":
        if byteorder != "little":
            raise RuntimeError("compact stream payloads require a little-endian host")
        if len(payload) % 8:
            raise ValueError("packed stream byte length must be a multiple of eight")
        result = cls()
        result.words.frombytes(payload)
        return result

    def __len__(self) -> int:
        return len(self.words)

    @overload
    def __getitem__(self, index: int) -> StreamBeat: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[StreamBeat, ...]: ...

    def __getitem__(self, index: int | slice) -> StreamBeat | tuple[StreamBeat, ...]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return tuple(StreamBeat(self.words[item], item == len(self) - 1) for item in indices)
        normalized = index + len(self) if index < 0 else index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        return StreamBeat(self.words[normalized], normalized == len(self) - 1)

    def __iter__(self) -> Iterator[StreamBeat]:
        final = len(self) - 1
        return (StreamBeat(word, index == final) for index, word in enumerate(self.words))


@dataclass(frozen=True)
class ProgramRecord:
    """One descriptor-program command and its stream contract."""

    command: Command128
    input_beats: Sequence[StreamBeat] = ()
    output_beats: int = 0

    def __post_init__(self) -> None:
        _unsigned("program output beat count", self.output_beats, 31)
        _unsigned("program input beat count", len(self.input_beats), 32)
        if not isinstance(self.input_beats, PackedStreamBeats):
            for index, beat in enumerate(self.input_beats):
                if beat.last != (index == len(self.input_beats) - 1):
                    raise ValueError("record input TLAST must mark only its final payload beat")


@dataclass(frozen=True)
class Command128:
    value: int

    def __post_init__(self) -> None:
        _unsigned("command", self.value, 128)

    @property
    def opcode(self) -> Opcode:
        return Opcode(self.value & 0xF)

    @property
    def words_le(self) -> tuple[int, int, int, int]:
        return tuple((self.value >> (32 * index)) & 0xFFFFFFFF for index in range(4))

    @classmethod
    def from_words_le(cls, words: Sequence[int]) -> "Command128":
        if len(words) != 4:
            raise ValueError("a command requires exactly four 32-bit words")
        value = 0
        for index, word in enumerate(words):
            value |= _unsigned(f"command word {index}", word, 32) << (32 * index)
        return cls(value)


def _unsigned(name: str, value: int, bits: int, maximum: int | None = None) -> int:
    integer = int(value)
    limit = (1 << bits) - 1 if maximum is None else maximum
    if integer < 0 or integer > limit:
        raise ValueError(f"{name} must be in [0,{limit}]")
    return integer


def _signed_bits(name: str, value: int, bits: int) -> int:
    integer = int(value)
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    if integer < minimum or integer > maximum:
        raise ValueError(f"{name} must be in [{minimum},{maximum}]")
    return integer & ((1 << bits) - 1)


def _decode_signed(value: int, bits: int) -> int:
    masked = value & ((1 << bits) - 1)
    return masked - (1 << bits) if masked & (1 << (bits - 1)) else masked


def pack_descriptor_program(
    records: Sequence[ProgramRecord],
) -> tuple[tuple[StreamBeat, ...], int]:
    """Pack records into the descriptor sequencer's single-DMA stream."""

    if not records:
        raise ValueError("a descriptor program must contain at least one record")
    words: list[int] = []
    total_outputs = 0
    for index, record in enumerate(records):
        final = index == len(records) - 1
        header = len(record.input_beats)
        header |= record.output_beats << 32
        header |= int(final) << 63
        words.extend((
            header,
            record.command.value & ((1 << 64) - 1),
            record.command.value >> 64,
        ))
        words.extend(beat.data for beat in record.input_beats)
        total_outputs += record.output_beats
        _unsigned("program total output beat count", total_outputs, 32)
    beats = tuple(StreamBeat(word, item == len(words) - 1)
                  for item, word in enumerate(words))
    return beats, total_outputs


def _put(command: int, value: int, offset: int, bits: int) -> int:
    return command | ((value & ((1 << bits) - 1)) << offset)


def clear_status_command() -> Command128:
    return Command128(int(Opcode.CLEAR_STATUS))


def linear_backward_command(
    rows: int,
    input_features: int,
    output_features: int,
    *,
    input_gradient_shift: int = 0,
    weight_gradient_shift: int = 0,
    mode: LinearGradientMode = LinearGradientMode.BOTH,
    accumulate: bool = False,
    suppress_output: bool = False,
) -> Command128:
    """Configure one signed 8x8 linear-layer backward tile."""

    row_count = _unsigned("rows", rows, 4, 8)
    input_count = _unsigned("input_features", input_features, 4, 8)
    output_count = _unsigned("output_features", output_features, 4, 8)
    if row_count == 0 or input_count == 0 or output_count == 0:
        raise ValueError("linear backward dimensions must be non-zero")
    command = _put(int(Opcode.TRAIN_LINEAR_BACKWARD), row_count, 4, 4)
    command = _put(command, input_count, 8, 4)
    command = _put(command, output_count, 12, 4)
    command = _put(
        command,
        _unsigned("input_gradient_shift", input_gradient_shift, 5),
        16,
        5,
    )
    command = _put(
        command,
        _unsigned("weight_gradient_shift", weight_gradient_shift, 5),
        21,
        5,
    )
    try:
        selected_mode = LinearGradientMode(mode)
    except ValueError as error:
        raise ValueError("linear backward mode must be BOTH, INPUT, or WEIGHT") from error
    command = _put(command, int(selected_mode), 26, 2)
    command = _put(command, int(bool(accumulate)), 28, 1)
    command = _put(command, int(bool(suppress_output)), 29, 1)
    return Command128(command)


def tensor_command(
    opcode: Opcode,
    *,
    rows: int = 0,
    columns: int = 0,
    issue_count: int,
    activation_zero_point: int = 0,
    weight_zero_point: int = 0,
) -> Command128:
    tensor_opcodes = {
        Opcode.LOAD_ACTIVATIONS,
        Opcode.LOAD_WEIGHTS_OUTER,
        Opcode.LOAD_WEIGHTS_DOT8,
        Opcode.COMPUTE_DENSE_OUTER,
        Opcode.COMPUTE_DENSE_DOT8,
        Opcode.COMPUTE_BINARY,
        Opcode.COMPUTE_TERNARY,
    }
    if opcode not in tensor_opcodes:
        raise ValueError("tensor_command requires a tensor opcode")
    maximum_issues = 256 if opcode in {Opcode.LOAD_WEIGHTS_DOT8, Opcode.COMPUTE_DENSE_DOT8} else 2048
    count = _unsigned("issue_count", issue_count, 12, maximum_issues)
    if count == 0:
        raise ValueError("issue_count must be non-zero")

    compute = opcode in {
        Opcode.COMPUTE_DENSE_OUTER,
        Opcode.COMPUTE_DENSE_DOT8,
        Opcode.COMPUTE_BINARY,
        Opcode.COMPUTE_TERNARY,
    }
    if compute:
        row_count = _unsigned("rows", rows, 4, 8)
        column_count = _unsigned("columns", columns, 6, 16)
        if row_count == 0 or column_count == 0:
            raise ValueError("compute rows and columns must be non-zero")
        if opcode == Opcode.COMPUTE_DENSE_DOT8 and row_count != 1:
            raise ValueError("dense Dot8 mode requires exactly one output row")
    else:
        row_count = _unsigned("rows", rows, 4, 8)
        column_count = _unsigned("columns", columns, 6, 16)

    command = int(opcode)
    command = _put(command, row_count, 4, 4)
    command = _put(command, column_count, 8, 6)
    command = _put(command, count, 14, 12)
    command = _put(command, _unsigned("activation_zero_point", activation_zero_point, 8), 26, 8)
    command = _put(command, _unsigned("weight_zero_point", weight_zero_point, 8), 34, 8)
    return Command128(command)


def alpha_xnor_command(active_bits: int, alpha_q8: int) -> Command128:
    active = _unsigned("active_bits", active_bits, 9, 256)
    alpha = _unsigned("alpha_q8", alpha_q8, 9, 256)
    if active == 0:
        raise ValueError("active_bits must be non-zero")
    command = _put(int(Opcode.ALPHA_XNOR), active, 42, 9)
    command = _put(command, alpha, 51, 9)
    return Command128(command)


def vector_command(
    operation: int,
    *,
    multiplier_q15: int = 0,
    right_shift: int = 0,
    output_zero_point: int = 0,
    threshold: int = 0,
    surrogate_window: int = 0,
) -> Command128:
    command = _put(int(Opcode.VECTOR_ALU), _unsigned("operation", operation, 4, 12), 4, 4)
    command = _put(command, _signed_bits("multiplier_q15", multiplier_q15, 16), 8, 16)
    if operation == 9:
        if right_shift == -16:
            raise ValueError("binary_point_shift must be in [-15,15]")
        shift_bits = _signed_bits("binary_point_shift", right_shift, 5)
    else:
        shift_bits = _unsigned("right_shift", right_shift, 5)
    command = _put(command, shift_bits, 24, 5)
    command = _put(command, _signed_bits("output_zero_point", output_zero_point, 16), 29, 16)
    command = _put(command, _signed_bits("threshold", threshold, 32), 45, 32)
    command = _put(command, _unsigned("surrogate_window", surrogate_window, 32), 77, 32)
    return Command128(command)


def layernorm_command(feature_count: int, epsilon: int = 1) -> Command128:
    count = _unsigned("feature_count", feature_count, 12, 2048)
    if count == 0:
        raise ValueError("feature_count must be non-zero")
    command = _put(int(Opcode.LAYERNORM), count, 4, 12)
    command = _put(command, _unsigned("epsilon", epsilon, 32), 16, 32)
    return Command128(command)


def log_softmax_command(item_count: int, difference_shift: int) -> Command128:
    count = _unsigned("item_count", item_count, 9, 256)
    if count == 0:
        raise ValueError("item_count must be non-zero")
    command = _put(int(Opcode.LOG_SOFTMAX), count, 4, 9)
    command = _put(command, _unsigned("difference_shift", difference_shift, 5), 13, 5)
    return Command128(command)


def lif_command(
    leak_shift: int,
    threshold: int,
    reset_value: int,
    *,
    clear_state: bool = False,
    trace_state: bool = False,
) -> Command128:
    command = _put(int(Opcode.LIF_STEP), _unsigned("leak_shift", leak_shift, 5), 4, 5)
    command = _put(command, _signed_bits("threshold", threshold, 24), 9, 24)
    command = _put(command, _signed_bits("reset_value", reset_value, 24), 33, 24)
    command = _put(command, int(bool(clear_state)), 57, 1)
    command = _put(command, int(bool(trace_state)), 58, 1)
    return Command128(command)


def sgd_command(
    learning_rate_q16: int,
    momentum_q16: int,
    weight_decay_q16: int,
    *,
    update_right_shift: int = 16,
) -> Command128:
    command = _put(int(Opcode.SGD_UPDATE), _unsigned("learning_rate_q16", learning_rate_q16, 18, 131071), 4, 18)
    command = _put(command, _unsigned("momentum_q16", momentum_q16, 17, 65536), 22, 17)
    command = _put(command, _unsigned("weight_decay_q16", weight_decay_q16, 17, 65536), 39, 17)
    command = _put(command, _unsigned("update_right_shift", update_right_shift, 5), 56, 5)
    return Command128(command)


def lut_command(suboperation: LutSuboperation, item_count: int = 0) -> Command128:
    count = _unsigned("item_count", item_count, 12, 256)
    if suboperation == LutSuboperation.LOAD and count == 0:
        raise ValueError("LUT load requires at least one entry")
    if suboperation == LutSuboperation.EVALUATE and count not in {0, 8}:
        raise ValueError("LUT evaluate processes exactly eight indices")
    if suboperation == LutSuboperation.CLEAR and count != 0:
        raise ValueError("LUT clear does not accept stream items")
    command = _put(int(Opcode.ACTIVATION_LUT), int(suboperation), 4, 4)
    command = _put(command, count, 8, 12)
    return Command128(command)


def _finish_beats(words: Iterable[int]) -> PackedStreamBeats:
    return PackedStreamBeats(words)


def pack_activation_beats(vectors: Sequence[Sequence[int]]) -> PackedStreamBeats:
    words = []
    for issue, vector in enumerate(vectors):
        if len(vector) != 8:
            raise ValueError(f"activation issue {issue} must contain eight bytes")
        word = sum(_unsigned(f"activation[{issue},{lane}]", value, 8) << (8 * lane)
                   for lane, value in enumerate(vector))
        words.append(word)
    return _finish_beats(words)


def pack_outer_weight_beats(weights: Sequence[Sequence[int]]) -> PackedStreamBeats:
    words = []
    for issue, vector in enumerate(weights):
        if len(vector) != 16:
            raise ValueError(f"outer weight issue {issue} must contain sixteen bytes")
        for half in range(2):
            words.append(sum(_unsigned(f"weight[{issue},{8 * half + lane}]", vector[8 * half + lane], 8)
                             << (8 * lane) for lane in range(8)))
    return _finish_beats(words)


def pack_dot8_weight_beats(weights: Sequence[Sequence[Sequence[int]]]) -> PackedStreamBeats:
    words = []
    for issue, matrix in enumerate(weights):
        if len(matrix) != 8 or any(len(row) != 16 for row in matrix):
            raise ValueError(f"Dot8 issue {issue} must have shape [8,16]")
        for reduction_lane, row in enumerate(matrix):
            for half in range(2):
                words.append(sum(_unsigned(
                    f"dot8_weight[{issue},{reduction_lane},{8 * half + lane}]",
                    row[8 * half + lane], 8) << (8 * lane) for lane in range(8)))
    return _finish_beats(words)


def pack_alpha_beats(query: Sequence[int], key: Sequence[int]) -> PackedStreamBeats:
    if len(query) != 256 or len(key) != 256:
        raise ValueError("alpha-XNOR query and key must each contain 256 bits")
    words = []
    for vector_name, vector in (("query", query), ("key", key)):
        for word_index in range(4):
            word = 0
            for bit_index in range(64):
                bit = _unsigned(f"{vector_name} bit", vector[word_index * 64 + bit_index], 1)
                word |= bit << bit_index
            words.append(word)
    return _finish_beats(words)


def pack_vector_beats(
    operand_a: Sequence[int],
    operand_b: Sequence[int],
    *,
    clamp_minimum: int = -(1 << 31),
    clamp_maximum: int = (1 << 31) - 1,
) -> PackedStreamBeats:
    if len(operand_a) != 8 or len(operand_b) != 8:
        raise ValueError("vector operands must each contain eight signed 32-bit lanes")
    words = [_signed_bits("clamp_minimum", clamp_minimum, 32)
             | (_signed_bits("clamp_maximum", clamp_maximum, 32) << 32)]
    for name, operand in (("operand_a", operand_a), ("operand_b", operand_b)):
        for lane in range(0, 8, 2):
            words.append(_signed_bits(f"{name}[{lane}]", operand[lane], 32)
                         | (_signed_bits(f"{name}[{lane + 1}]", operand[lane + 1], 32) << 32))
    return _finish_beats(words)


def pack_layernorm_beats(
    values: Sequence[int], gamma_q15: Sequence[int], beta: Sequence[int]
) -> PackedStreamBeats:
    if not (len(values) == len(gamma_q15) == len(beta)) or len(values) == 0:
        raise ValueError("LayerNorm value, gamma, and beta arrays must have equal non-zero length")
    words = []
    for index, (value, gamma, offset) in enumerate(zip(values, gamma_q15, beta)):
        beta_bits = _signed_bits(f"beta[{index}]", offset, 24)
        word0 = _signed_bits(f"value[{index}]", value, 24)
        word0 |= _signed_bits(f"gamma[{index}]", gamma, 24) << 24
        word0 |= (beta_bits & 0xFFFF) << 48
        words.extend((word0, beta_bits >> 16))
    return _finish_beats(words)


def pack_softmax_beats(scores: Sequence[int]) -> PackedStreamBeats:
    if not scores:
        raise ValueError("softmax requires at least one score")
    return _finish_beats(_signed_bits(f"score[{index}]", score, 24)
                         for index, score in enumerate(scores))


def pack_lif_beats(currents: Sequence[int]) -> PackedStreamBeats:
    if len(currents) != 32:
        raise ValueError("LIF current vector must contain 32 signed 16-bit lanes")
    words = []
    for lane in range(0, 32, 4):
        words.append(sum(_signed_bits(f"current[{lane + offset}]", currents[lane + offset], 16)
                         << (16 * offset) for offset in range(4)))
    return _finish_beats(words)


def pack_optimizer_beats(
    weights: Sequence[int], gradients: Sequence[int], velocities: Sequence[int]
) -> PackedStreamBeats:
    if not (len(weights) == len(gradients) == len(velocities) == 8):
        raise ValueError("optimizer vectors must each contain eight lanes")
    words = []
    for lane in range(8):
        word = _signed_bits(f"weight[{lane}]", weights[lane], 16)
        word |= _signed_bits(f"gradient[{lane}]", gradients[lane], 24) << 16
        word |= _signed_bits(f"velocity[{lane}]", velocities[lane], 24) << 40
        words.append(word)
    return _finish_beats(words)


def pack_linear_backward_beats(
    activations: Sequence[Sequence[int]],
    weights: Sequence[Sequence[int]],
    output_gradients: Sequence[Sequence[int]],
    *,
    mode: LinearGradientMode = LinearGradientMode.BOTH,
) -> PackedStreamBeats:
    """Pack signed X, W, and dY matrices for one linear backward tile."""

    rows = len(activations)
    input_features = len(weights)
    output_features = len(weights[0]) if input_features else 0
    if not 1 <= rows <= 8 or not 1 <= input_features <= 8 or not 1 <= output_features <= 8:
        raise ValueError("linear backward tile dimensions must be in [1,8]")
    if any(len(row) != input_features for row in activations):
        raise ValueError("activation rows must match the weight input dimension")
    if any(len(row) != output_features for row in weights):
        raise ValueError("weight rows must have one common output dimension")
    if len(output_gradients) != rows or any(
        len(row) != output_features for row in output_gradients
    ):
        raise ValueError("output gradients must have shape [rows, output_features]")

    try:
        selected_mode = LinearGradientMode(mode)
    except ValueError as error:
        raise ValueError("linear backward mode must be BOTH, INPUT, or WEIGHT") from error
    words: list[int] = []
    if selected_mode != LinearGradientMode.INPUT:
        for row_index, row in enumerate(activations):
            padded = list(row) + [0] * (8 - input_features)
            for half in range(2):
                words.append(sum(
                    _signed_bits(
                        f"activation[{row_index},{half * 4 + lane}]",
                        padded[half * 4 + lane],
                        16,
                    ) << (16 * lane)
                    for lane in range(4)
                ))
    if selected_mode != LinearGradientMode.WEIGHT:
        for row_index, row in enumerate(weights):
            padded = list(row) + [0] * (8 - output_features)
            for half in range(2):
                words.append(sum(
                    _signed_bits(
                        f"weight[{row_index},{half * 4 + lane}]",
                        padded[half * 4 + lane],
                        16,
                    ) << (16 * lane)
                    for lane in range(4)
                ))
    for row_index, row in enumerate(output_gradients):
        padded = list(row) + [0] * (8 - output_features)
        for half in range(2):
            words.append(sum(
                _signed_bits(
                    f"output_gradient[{row_index},{half * 4 + lane}]",
                    padded[half * 4 + lane],
                    16,
                ) << (16 * lane)
                for lane in range(4)
            ))
    return _finish_beats(words)


def pack_lut_load_beats(entries: Sequence[tuple[int, int]]) -> PackedStreamBeats:
    if not entries:
        raise ValueError("LUT load requires at least one entry")
    words = []
    for index, (address, value) in enumerate(entries):
        words.append(_unsigned(f"LUT address {index}", address, 8)
                     | (_signed_bits(f"LUT value {index}", value, 16) << 8))
    return _finish_beats(words)


def pack_lut_eval_beat(indices: Sequence[int]) -> PackedStreamBeats:
    if len(indices) != 8:
        raise ValueError("LUT evaluate requires exactly eight indices")
    word = sum(_unsigned(f"LUT index {lane}", value, 8) << (8 * lane)
               for lane, value in enumerate(indices))
    return [StreamBeat(word, True)]


def unpack_tensor_result(word: int) -> tuple[int, int, int, bool]:
    value = _unsigned("tensor result word", word, 64)
    return (_decode_signed(value, 48), (value >> 48) & 0xF,
            (value >> 52) & 0x3F, bool((value >> 58) & 1))


def unpack_alpha_result(word: int) -> tuple[int, int, int]:
    value = _unsigned("alpha result word", word, 64)
    return value & 0x3FFFF, (value >> 18) & 0x1FF, (value >> 27) & 0x1FF


def unpack_vector_result(words: Sequence[int]) -> tuple[int, ...]:
    if len(words) != 4:
        raise ValueError("vector result requires four words")
    return tuple(_decode_signed(word >> shift, 32)
                 for word in words for shift in (0, 32))


def unpack_layernorm_result(word: int) -> tuple[int, int, bool]:
    value = _unsigned("LayerNorm result word", word, 64)
    return _decode_signed(value, 32), (value >> 32) & 0xFFF, bool((value >> 44) & 1)


def unpack_softmax_result(word: int) -> tuple[int, int, bool]:
    value = _unsigned("softmax result word", word, 64)
    return value & 0xFFFF, (value >> 16) & 0x1FF, bool((value >> 25) & 1)


def unpack_lif_result(word: int) -> tuple[int, int]:
    value = _unsigned("LIF result word", word, 64)
    return value & 0xFFFFFFFF, value >> 32


def unpack_lif_trace(words: Sequence[int]) -> tuple[int, int, tuple[int, ...]]:
    """Decode one traced LIF response: metadata followed by 32 s24 membranes."""

    if len(words) != 13:
        raise ValueError("a traced LIF result requires exactly thirteen words")
    spikes, timestep = unpack_lif_result(words[0])
    packed_membranes = 0
    for index, word in enumerate(words[1:]):
        packed_membranes |= _unsigned(f"LIF trace word {index}", word, 64) << (64 * index)
    membranes = tuple(
        _decode_signed(packed_membranes >> (24 * lane), 24)
        for lane in range(32)
    )
    return spikes, timestep, membranes


def unpack_optimizer_result(words: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(words) != 8:
        raise ValueError("optimizer result requires eight words")
    weights = tuple(_decode_signed(word, 16) for word in words)
    velocities = tuple(_decode_signed(word >> 16, 24) for word in words)
    return weights, velocities


def unpack_linear_backward_result(
    words: Sequence[int],
    rows: int,
    input_features: int,
    output_features: int,
    *,
    mode: LinearGradientMode = LinearGradientMode.BOTH,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Decode padded s32 dX rows and packed s24 dW rows from the RTL."""

    if not 1 <= rows <= 8 or not 1 <= input_features <= 8 or not 1 <= output_features <= 8:
        raise ValueError("linear backward tile dimensions must be in [1,8]")
    try:
        selected_mode = LinearGradientMode(mode)
    except ValueError as error:
        raise ValueError("linear backward mode must be BOTH, INPUT, or WEIGHT") from error
    expected_words = 0
    if selected_mode != LinearGradientMode.WEIGHT:
        expected_words += 4 * rows
    if selected_mode != LinearGradientMode.INPUT:
        expected_words += 3 * input_features
    if len(words) != expected_words:
        raise ValueError(
            f"linear backward result requires {expected_words} words; received {len(words)}"
        )
    input_gradients: list[tuple[int, ...]] = []
    cursor = 0
    if selected_mode != LinearGradientMode.WEIGHT:
        for _ in range(rows):
            lanes = tuple(
                _decode_signed(words[cursor + lane // 2] >> (32 * (lane % 2)), 32)
                for lane in range(input_features)
            )
            input_gradients.append(lanes)
            cursor += 4
    weight_gradients: list[tuple[int, ...]] = []
    if selected_mode != LinearGradientMode.INPUT:
        for _ in range(input_features):
            packed = sum(words[cursor + index] << (64 * index) for index in range(3))
            weight_gradients.append(tuple(
                _decode_signed(packed >> (24 * lane), 24)
                for lane in range(output_features)
            ))
            cursor += 3
    return tuple(input_gradients), tuple(weight_gradients)


def unpack_lut_result(words: Sequence[int]) -> tuple[int, ...]:
    if len(words) != 2:
        raise ValueError("LUT result requires two words")
    return tuple(_decode_signed(word >> (16 * lane), 16)
                 for word in words for lane in range(4))
