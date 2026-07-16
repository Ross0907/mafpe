from __future__ import annotations

import unittest

import numpy as np

from neuraedge_compiler import (
    compile_spike_conv2d,
    compile_spike_gemm,
    compile_u8_conv2d,
    compile_u8_gemm,
)
from neuraedge_golden import Dataflow, direct_centered_gemm
from neuraedge_protocol import Command128, Opcode, StreamBeat


def _decode_bytes(word: int) -> list[int]:
    return [(word >> (8 * lane)) & 0xFF for lane in range(8)]


class TensorTransactionEmulator:
    """Independent stateful interpretation of the public tensor stream ABI."""

    def __init__(self) -> None:
        self.activations: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    def execute(
        self,
        command: Command128,
        input_beats: tuple[StreamBeat, ...] = (),
        *,
        output_beats: int = 0,
        timeout_seconds: float = 5.0,
    ) -> tuple[int, ...]:
        del timeout_seconds
        value = command.value
        opcode = command.opcode
        issues = (value >> 14) & 0xFFF

        if opcode == Opcode.LOAD_ACTIVATIONS:
            self.activations = np.asarray([_decode_bytes(beat.data) for beat in input_beats], dtype=np.int64)
            self.assert_shape(self.activations, (issues, 8))
            return ()
        if opcode == Opcode.LOAD_WEIGHTS_OUTER:
            words = [_decode_bytes(beat.data) for beat in input_beats]
            self.weights = np.asarray(
                [words[2 * issue] + words[2 * issue + 1] for issue in range(issues)],
                dtype=np.int64,
            )
            return ()
        if opcode == Opcode.LOAD_WEIGHTS_DOT8:
            words = [_decode_bytes(beat.data) for beat in input_beats]
            self.weights = np.asarray([
                [words[16 * issue + 2 * lane] + words[16 * issue + 2 * lane + 1]
                 for lane in range(8)]
                for issue in range(issues)
            ], dtype=np.int64)
            return ()

        if self.activations is None or self.weights is None:
            raise AssertionError("compute was issued before operand loads")
        rows = (value >> 4) & 0xF
        columns = (value >> 8) & 0x3F
        za = (value >> 26) & 0xFF
        zb = (value >> 34) & 0xFF
        if opcode == Opcode.COMPUTE_DENSE_OUTER:
            result = (self.activations[:, :rows].T - za) @ (self.weights[:, :columns] - zb)
        elif opcode == Opcode.COMPUTE_DENSE_DOT8:
            lhs = self.activations.reshape(-1) - za
            rhs = self.weights.reshape(-1, 16)[:, :columns] - zb
            result = (lhs @ rhs).reshape(1, columns)
        elif opcode in {Opcode.COMPUTE_BINARY, Opcode.COMPUTE_TERNARY}:
            lhs = self.activations[:, :rows].T.copy()
            if opcode == Opcode.COMPUTE_TERNARY:
                lhs[lhs >= 128] -= 256
            rhs = self.weights[:, :columns].copy()
            rhs[rhs >= 128] -= 256
            result = lhs @ rhs
        else:
            raise AssertionError(f"unexpected opcode {opcode}")

        words = []
        for row in range(rows):
            for column in range(columns):
                encoded = int(result[row, column]) & ((1 << 48) - 1)
                encoded |= row << 48
                encoded |= column << 52
                encoded |= int(len(words) == output_beats - 1) << 58
                words.append(encoded)
        if len(words) != output_beats:
            raise AssertionError("compiler requested the wrong output beat count")
        return tuple(words)

    def execute_program(
        self,
        records,
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, ...]:
        words: list[int] = []
        for record in records:
            words.extend(self.execute(
                record.command,
                record.input_beats,
                output_beats=record.output_beats,
                timeout_seconds=timeout_seconds,
            ))
        return tuple(words)

    @staticmethod
    def assert_shape(array: np.ndarray, shape: tuple[int, ...]) -> None:
        if array.shape != shape:
            raise AssertionError(f"stream shape {array.shape} != {shape}")


def direct_conv2d(
    activations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    activation_zero_point: int = 0,
    weight_zero_point: int = 0,
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    dilation: tuple[int, int] = (1, 1),
    groups: int = 1,
) -> np.ndarray:
    batch, input_channels, height, width = activations.shape
    output_channels, channels_per_group, kernel_height, kernel_width = weights.shape
    outputs_per_group = output_channels // groups
    output_height = (height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0] + 1
    output_width = (width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1] + 1
    output = np.zeros((batch, output_channels, output_height, output_width), dtype=np.int64)
    for n in range(batch):
        for output_channel in range(output_channels):
            group = output_channel // outputs_per_group
            input_start = group * channels_per_group
            for output_y in range(output_height):
                for output_x in range(output_width):
                    accumulator = int(bias[output_channel])
                    for local_channel in range(channels_per_group):
                        input_channel = input_start + local_channel
                        for kernel_y in range(kernel_height):
                            input_y = output_y * stride[0] + kernel_y * dilation[0] - padding[0]
                            for kernel_x in range(kernel_width):
                                input_x = output_x * stride[1] + kernel_x * dilation[1] - padding[1]
                                activation = activation_zero_point
                                if 0 <= input_y < height and 0 <= input_x < width:
                                    activation = int(activations[n, input_channel, input_y, input_x])
                                weight = int(weights[output_channel, local_channel, kernel_y, kernel_x])
                                accumulator += ((activation - activation_zero_point)
                                                * (weight - weight_zero_point))
                    output[n, output_channel, output_y, output_x] = accumulator
    return output


class CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260713)

    def check_case(self, rows: int, reduction: int, columns: int, dataflow: Dataflow | None = None) -> None:
        a = self.rng.integers(0, 256, size=(rows, reduction), dtype=np.uint8)
        b = self.rng.integers(0, 256, size=(reduction, columns), dtype=np.uint8)
        bias = self.rng.integers(-100000, 100001, size=columns, dtype=np.int64)
        za = int(self.rng.integers(0, 256))
        zb = int(self.rng.integers(0, 256))
        program = compile_u8_gemm(
            a, b, activation_zero_point=za, weight_zero_point=zb, bias=bias,
            dataflow=dataflow,
        )
        actual = program.execute(TensorTransactionEmulator())
        batched = program.execute_batched(TensorTransactionEmulator())
        expected = direct_centered_gemm(a, b, za, zb, bias)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(batched, expected)
        self.assertEqual(program.mathematical_macs, rows * reduction * columns)
        self.assertGreater(program.estimated_input_bytes, 0)

    def test_outer_mode_irregular_tiles_both_dataflows(self) -> None:
        self.check_case(13, 37, 29, Dataflow.INPUT_BROADCAST)
        self.check_case(17, 33, 19, Dataflow.WEIGHT_BROADCAST)

    def test_dot8_single_row_with_padding(self) -> None:
        self.check_case(1, 19, 35)

    def test_reduction_larger_than_local_memory(self) -> None:
        self.check_case(9, 2053, 17)

    def test_zero_skipping_and_boundary_values(self) -> None:
        a = np.full((8, 9), 128, dtype=np.uint8)
        b = np.full((9, 16), 129, dtype=np.uint8)
        a[3, 4] = 255
        b[4, 7] = 0
        program = compile_u8_gemm(a, b, activation_zero_point=128, weight_zero_point=129)
        np.testing.assert_array_equal(
            program.execute(TensorTransactionEmulator()),
            direct_centered_gemm(a, b, 128, 129),
        )

    def test_binary_and_ternary_sparse_paths(self) -> None:
        weights = self.rng.integers(-128, 128, size=(41, 23), dtype=np.int16)
        binary = self.rng.integers(0, 2, size=(11, 41), dtype=np.int8)
        ternary = self.rng.integers(-1, 2, size=(9, 41), dtype=np.int8)
        binary_program = compile_spike_gemm(binary, weights)
        ternary_program = compile_spike_gemm(
            ternary, weights, ternary=True, dataflow=Dataflow.WEIGHT_BROADCAST,
        )
        np.testing.assert_array_equal(
            binary_program.execute(TensorTransactionEmulator()),
            binary.astype(np.int64) @ weights.astype(np.int64),
        )
        np.testing.assert_array_equal(
            ternary_program.execute(TensorTransactionEmulator()),
            ternary.astype(np.int64) @ weights.astype(np.int64),
        )
        self.assertEqual(binary_program.estimated_spike_adds, int(np.count_nonzero(binary)) * 23)
        self.assertEqual(ternary_program.estimated_spike_adds, int(np.count_nonzero(ternary)) * 23)

    def test_grouped_u8_convolution_with_stride_padding_and_dilation(self) -> None:
        activations = self.rng.integers(0, 256, size=(1, 4, 5, 6), dtype=np.uint8)
        weights = self.rng.integers(0, 256, size=(6, 2, 2, 3), dtype=np.uint8)
        bias = self.rng.integers(-5000, 5001, size=6, dtype=np.int64)
        program = compile_u8_conv2d(
            activations,
            weights,
            activation_zero_point=117,
            weight_zero_point=133,
            bias=bias,
            stride=(2, 1),
            padding=(1, 2),
            dilation=(2, 1),
            groups=2,
        )
        actual = program.execute(TensorTransactionEmulator())
        expected = direct_conv2d(
            activations, weights, bias,
            activation_zero_point=117, weight_zero_point=133,
            stride=(2, 1), padding=(1, 2), dilation=(2, 1), groups=2,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (1, 6, 3, 8))

    def test_depthwise_binary_and_ternary_spike_convolution(self) -> None:
        binary = self.rng.integers(0, 2, size=(1, 3, 4, 4), dtype=np.int8)
        ternary = self.rng.integers(-1, 2, size=(1, 3, 4, 4), dtype=np.int8)
        weights = self.rng.integers(-128, 128, size=(3, 1, 3, 3), dtype=np.int16)
        bias = self.rng.integers(-50, 51, size=3, dtype=np.int64)
        for spikes, is_ternary in ((binary, False), (ternary, True)):
            program = compile_spike_conv2d(
                spikes, weights, ternary=is_ternary, bias=bias,
                padding=1, groups=3,
            )
            actual = program.execute(TensorTransactionEmulator())
            expected = direct_conv2d(
                spikes, weights, bias, padding=(1, 1), groups=3,
            )
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(program.estimated_spike_adds,
                             sum(group.gemm.estimated_spike_adds for group in program.group_programs))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            compile_u8_gemm([[1, 2]], [[1, 2]])
        with self.assertRaises(ValueError):
            compile_u8_gemm([[256]], [[1]])
        with self.assertRaises(ValueError):
            compile_u8_gemm([[1]], [[1]], bias=[1, 2])
        with self.assertRaises(ValueError):
            compile_spike_gemm([[0, 2]], [[1], [2]])
        with self.assertRaises(ValueError):
            compile_spike_gemm([[0, -1]], [[1], [2]], ternary=False)
        with self.assertRaises(ValueError):
            compile_u8_conv2d(np.zeros((1, 3, 4, 4), dtype=np.uint8),
                              np.zeros((4, 2, 3, 3), dtype=np.uint8), groups=2)


if __name__ == "__main__":
    unittest.main()
