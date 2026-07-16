from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neuraedge_golden import (
    VectorOperation,
    log_int_softmax_q15,
    round_shift_signed,
    saturate_signed,
    sgd_momentum_step,
    vector_alu,
)
from neuraedge_protocol import LinearGradientMode, Opcode
from neuraedge_training import (
    compile_affine_backward,
    compile_conv2d_backward,
    compile_linear_backward,
    build_spikingformer_training_coverage,
    execute_lif_bptt,
    execute_int8_weight_projection,
    execute_maxpool2d_backward,
    execute_sgdw_update,
    lif_bptt_fixed,
    maxpool2d_backward_fixed,
    sigmoid_surrogate_lut_q15,
    surrogate_lut_addresses,
)
from tests.test_neuraedge_compiler import TensorTransactionEmulator


def _signed(value: int, bits: int) -> int:
    masked = value & ((1 << bits) - 1)
    return masked - (1 << bits) if masked & (1 << (bits - 1)) else masked


class BackwardRtlEmulator:
    """Independent command-level model of retained raw gradient sums."""

    def __init__(self) -> None:
        self.input_accumulator: np.ndarray | None = None
        self.weight_accumulator: np.ndarray | None = None

    @staticmethod
    def _round_and_clip(values: np.ndarray, shift: int, bits: int) -> np.ndarray:
        return saturate_signed(round_shift_signed(values, shift), bits)

    @staticmethod
    def _decode(command, beats):
        value = command.value
        rows = (value >> 4) & 0xF
        inputs = (value >> 8) & 0xF
        outputs = (value >> 12) & 0xF
        mode = LinearGradientMode((value >> 26) & 0x3)
        words = [beat.data for beat in beats]
        cursor = 0
        x = np.zeros((rows, inputs), dtype=np.int64)
        w = np.zeros((inputs, outputs), dtype=np.int64)
        if mode != LinearGradientMode.INPUT:
            for row in range(rows):
                packed = words[cursor] | (words[cursor + 1] << 64)
                cursor += 2
                for lane in range(inputs):
                    x[row, lane] = _signed(packed >> (16 * lane), 16)
        if mode != LinearGradientMode.WEIGHT:
            for row in range(inputs):
                packed = words[cursor] | (words[cursor + 1] << 64)
                cursor += 2
                for lane in range(outputs):
                    w[row, lane] = _signed(packed >> (16 * lane), 16)
        dy = np.zeros((rows, outputs), dtype=np.int64)
        for row in range(rows):
            packed = words[cursor] | (words[cursor + 1] << 64)
            cursor += 2
            for lane in range(outputs):
                dy[row, lane] = _signed(packed >> (16 * lane), 16)
        if cursor != len(words):
            raise AssertionError("compiler emitted an unexpected backward stream length")
        return rows, inputs, outputs, mode, x, w, dy

    @staticmethod
    def _pack_dx(values: np.ndarray) -> tuple[int, ...]:
        words: list[int] = []
        for row in values:
            padded = list(map(int, row)) + [0] * (8 - len(row))
            for lane in range(0, 8, 2):
                words.append(
                    (padded[lane] & 0xFFFFFFFF)
                    | ((padded[lane + 1] & 0xFFFFFFFF) << 32)
                )
        return tuple(words)

    @staticmethod
    def _pack_dw(values: np.ndarray) -> tuple[int, ...]:
        words: list[int] = []
        for row in values:
            padded = list(map(int, row)) + [0] * (8 - len(row))
            packed = sum(
                (item & 0xFFFFFF) << (24 * lane)
                for lane, item in enumerate(padded)
            )
            words.extend(
                (packed >> (64 * word)) & 0xFFFFFFFFFFFFFFFF for word in range(3)
            )
        return tuple(words)

    def execute(
        self,
        command,
        input_beats=(),
        *,
        output_beats=0,
        timeout_seconds=5.0,
    ) -> tuple[int, ...]:
        del timeout_seconds
        rows, inputs, outputs, mode, x, w, dy = self._decode(command, input_beats)
        accumulate = bool((command.value >> 28) & 1)
        suppress = bool((command.value >> 29) & 1)
        dx_shift = (command.value >> 16) & 0x1F
        dw_shift = (command.value >> 21) & 0x1F
        result: tuple[int, ...]
        if mode == LinearGradientMode.INPUT:
            partial = np.zeros((rows, inputs), dtype=np.int64)
            for row in range(rows):
                for input_index in range(inputs):
                    partial[row, input_index] = sum(
                        int(dy[row, output]) * int(w[input_index, output])
                        for output in range(outputs)
                    )
            if accumulate:
                if self.input_accumulator is None:
                    raise AssertionError("input accumulation began without retained state")
                self.input_accumulator += partial
            else:
                self.input_accumulator = partial
            if suppress:
                result = ()
            else:
                result = self._pack_dx(
                    self._round_and_clip(self.input_accumulator, dx_shift, 32)
                )
                self.input_accumulator = None
        elif mode == LinearGradientMode.WEIGHT:
            partial = np.zeros((inputs, outputs), dtype=np.int64)
            for input_index in range(inputs):
                for output in range(outputs):
                    partial[input_index, output] = sum(
                        int(x[row, input_index]) * int(dy[row, output])
                        for row in range(rows)
                    )
            if accumulate:
                if self.weight_accumulator is None:
                    raise AssertionError("weight accumulation began without retained state")
                self.weight_accumulator += partial
            else:
                self.weight_accumulator = partial
            if suppress:
                result = ()
            else:
                result = self._pack_dw(
                    self._round_and_clip(self.weight_accumulator, dw_shift, 24)
                )
                self.weight_accumulator = None
        else:
            raise AssertionError("the tiled compiler must use one-gradient passes")
        if len(result) != output_beats:
            raise AssertionError(f"emulator produced {len(result)} words, expected {output_beats}")
        return result

    def execute_program(self, records, *, timeout_seconds=30.0) -> tuple[int, ...]:
        del timeout_seconds
        output: list[int] = []
        for record in records:
            output.extend(self.execute(
                record.command,
                record.input_beats,
                output_beats=record.output_beats,
            ))
        return tuple(output)


class FabricOperatorEmulator:
    def __init__(self) -> None:
        self.lut = np.zeros(256, dtype=np.int64)
        self.lif_membrane = np.zeros(32, dtype=np.int64)
        self.lif_timestep = 0
        self.command_count = 0

    def execute(
        self,
        command,
        input_beats=(),
        *,
        output_beats=0,
        timeout_seconds=5.0,
    ) -> tuple[int, ...]:
        del timeout_seconds
        self.command_count += 1
        words = [beat.data for beat in input_beats]
        if command.opcode == Opcode.ACTIVATION_LUT:
            suboperation = (command.value >> 4) & 0xF
            if suboperation == 1:
                for word in words:
                    self.lut[word & 0xFF] = _signed(word >> 8, 16)
                result: tuple[int, ...] = ()
            elif suboperation == 2:
                indices = [(words[0] >> (8 * lane)) & 0xFF for lane in range(8)]
                values = [int(self.lut[index]) for index in indices]
                result = tuple(
                    sum((values[word * 4 + lane] & 0xFFFF) << (16 * lane)
                        for lane in range(4))
                    for word in range(2)
                )
            else:
                self.lut.fill(0)
                result = ()
        elif command.opcode == Opcode.VECTOR_ALU:
            operation = VectorOperation((command.value >> 4) & 0xF)
            multiplier = _signed(command.value >> 8, 16)
            shift_bits = (command.value >> 24) & 0x1F
            right_shift = _signed(shift_bits, 5) if operation == VectorOperation.FLEX_REQUANTIZE else shift_bits
            output_zero_point = _signed(command.value >> 29, 16)
            threshold = _signed(command.value >> 45, 32)
            surrogate_window = (command.value >> 77) & 0xFFFFFFFF
            clamp_minimum = _signed(words[0], 32)
            clamp_maximum = _signed(words[0] >> 32, 32)
            operand_a = np.array([
                _signed(words[1 + lane // 2] >> (32 * (lane % 2)), 32)
                for lane in range(8)
            ], dtype=np.int64)
            operand_b = np.array([
                _signed(words[5 + lane // 2] >> (32 * (lane % 2)), 32)
                for lane in range(8)
            ], dtype=np.int64)
            values = vector_alu(
                operation,
                operand_a,
                operand_b,
                multiplier_q15=multiplier,
                right_shift=right_shift,
                output_zero_point=output_zero_point,
                threshold=threshold,
                surrogate_window=surrogate_window,
                clamp_minimum=clamp_minimum,
                clamp_maximum=clamp_maximum,
            )
            result = tuple(
                (int(values[lane]) & 0xFFFFFFFF)
                | ((int(values[lane + 1]) & 0xFFFFFFFF) << 32)
                for lane in range(0, 8, 2)
            )
        elif command.opcode == Opcode.SGD_UPDATE:
            weights = np.array([_signed(word, 16) for word in words], dtype=np.int64)
            gradients = np.array([_signed(word >> 16, 24) for word in words], dtype=np.int64)
            velocities = np.array([_signed(word >> 40, 24) for word in words], dtype=np.int64)
            learning_rate = (command.value >> 4) & 0x3FFFF
            momentum = (command.value >> 22) & 0x1FFFF
            weight_decay = (command.value >> 39) & 0x1FFFF
            update_right_shift = (command.value >> 56) & 0x1F
            updated_weight, updated_velocity = sgd_momentum_step(
                weights,
                gradients,
                velocities,
                learning_rate,
                momentum,
                weight_decay,
                update_right_shift,
            )
            result = tuple(
                (int(updated_weight[lane]) & 0xFFFF)
                | ((int(updated_velocity[lane]) & 0xFFFFFF) << 16)
                for lane in range(8)
            )
        elif command.opcode == Opcode.LOG_SOFTMAX:
            count = (command.value >> 4) & 0x1FF
            difference_shift = (command.value >> 13) & 0x1F
            scores = np.asarray(
                [_signed(word, 24) for word in words[:count]], dtype=np.int64)
            probabilities = log_int_softmax_q15(scores, difference_shift)
            result = tuple(
                int(probabilities[index])
                | (index << 16)
                | (int(index == count - 1) << 25)
                for index in range(count)
            )
        elif command.opcode == Opcode.LIF_STEP:
            leak_shift = (command.value >> 4) & 0x1F
            threshold = _signed(command.value >> 9, 24)
            reset = _signed(command.value >> 33, 24)
            if (command.value >> 57) & 1:
                self.lif_membrane.fill(reset)
                self.lif_timestep = 0
            currents = np.asarray([
                _signed(word >> (16 * lane), 16)
                for word in words for lane in range(4)
            ], dtype=np.int64)
            self.lif_membrane = saturate_signed(
                self.lif_membrane - (self.lif_membrane >> leak_shift) + currents,
                24,
            )
            pre_reset = self.lif_membrane.copy()
            fired = pre_reset >= threshold
            self.lif_membrane = np.where(fired, reset, pre_reset)
            self.lif_timestep += 1
            spike_word = sum(int(fired[lane]) << lane for lane in range(32))
            output = [spike_word | (self.lif_timestep << 32)]
            if (command.value >> 58) & 1:
                packed = sum(
                    (int(pre_reset[lane]) & 0xFFFFFF) << (24 * lane)
                    for lane in range(32)
                )
                output.extend(
                    (packed >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
                    for word in range(12)
                )
            result = tuple(output)
        elif command.opcode == Opcode.CLEAR_STATUS:
            result = ()
        else:
            raise AssertionError(f"unexpected fabric opcode {command.opcode}")
        if len(result) != output_beats:
            raise AssertionError(f"fabric emulator returned {len(result)} words, expected {output_beats}")
        return result


class FullTrainingEmulator:
    def __init__(self) -> None:
        self.backward = BackwardRtlEmulator()
        self.fabric = FabricOperatorEmulator()
        self.tensor = TensorTransactionEmulator()

    def execute(self, command, input_beats=(), *, output_beats=0, timeout_seconds=5.0):
        if command.opcode == Opcode.TRAIN_LINEAR_BACKWARD:
            return self.backward.execute(
                command,
                input_beats,
                output_beats=output_beats,
                timeout_seconds=timeout_seconds,
            )
        if command.opcode in {
            Opcode.LOAD_ACTIVATIONS,
            Opcode.LOAD_WEIGHTS_OUTER,
            Opcode.LOAD_WEIGHTS_DOT8,
            Opcode.COMPUTE_DENSE_OUTER,
            Opcode.COMPUTE_DENSE_DOT8,
            Opcode.COMPUTE_BINARY,
            Opcode.COMPUTE_TERNARY,
        }:
            return self.tensor.execute(
                command,
                input_beats,
                output_beats=output_beats,
                timeout_seconds=timeout_seconds,
            )
        return self.fabric.execute(
            command,
            input_beats,
            output_beats=output_beats,
            timeout_seconds=timeout_seconds,
        )

    def execute_program(self, records, *, timeout_seconds=30.0) -> tuple[int, ...]:
        output: list[int] = []
        for record in records:
            output.extend(self.execute(
                record.command,
                record.input_beats,
                output_beats=record.output_beats,
                timeout_seconds=timeout_seconds,
            ))
        return tuple(output)


class TrainingCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(20260713)

    def test_arbitrary_linear_shape_uses_exact_two_pass_accumulation(self) -> None:
        x = self.rng.integers(-128, 128, size=(13, 11), dtype=np.int64)
        w = self.rng.integers(-32768, 32768, size=(11, 17), dtype=np.int64)
        dy = self.rng.integers(-32768, 32768, size=(13, 17), dtype=np.int64)
        program = compile_linear_backward(
            x,
            w,
            dy,
            input_gradient_shift=4,
            weight_gradient_shift=3,
        )
        self.assertEqual(program.shape, (13, 11, 17))
        self.assertEqual(program.partial_tile_count, 24)
        self.assertEqual(program.mathematical_multiplications, 2 * 13 * 11 * 17)
        self.assertEqual(sum(
            instruction.mode == LinearGradientMode.INPUT
            for instruction in program.instructions
        ), 12)
        self.assertEqual(sum(
            instruction.mode == LinearGradientMode.WEIGHT
            for instruction in program.instructions
        ), 12)
        for instruction in program.instructions:
            self.assertEqual(
                instruction.record.output_beats == 0,
                instruction.suppress_output,
            )
            self.assertEqual(
                bool((instruction.record.command.value >> 28) & 1),
                instruction.accumulate,
            )

        expected_dx = saturate_signed(round_shift_signed(dy @ w.T, 4), 32)
        expected_dw = saturate_signed(round_shift_signed(x.T @ dy, 3), 24)
        got_dx, got_dw = program.execute(BackwardRtlEmulator())
        np.testing.assert_array_equal(got_dx, expected_dx)
        np.testing.assert_array_equal(got_dw, expected_dw)
        got_dx, got_dw = program.execute_batched(BackwardRtlEmulator())
        np.testing.assert_array_equal(got_dx, expected_dx)
        np.testing.assert_array_equal(got_dw, expected_dw)

    def test_affine_augmentation_returns_exact_bias_gradient(self) -> None:
        x = self.rng.integers(-32, 33, size=(9, 10), dtype=np.int64)
        w = self.rng.integers(-2000, 2001, size=(10, 7), dtype=np.int64)
        bias = self.rng.integers(-3000, 3001, size=7, dtype=np.int64)
        dy = self.rng.integers(-1000, 1001, size=(9, 7), dtype=np.int64)
        program = compile_affine_backward(
            x,
            w,
            bias,
            dy,
            input_gradient_shift=2,
            weight_gradient_shift=1,
        )
        got_dx, got_dw, got_db = program.execute_batched(BackwardRtlEmulator())
        np.testing.assert_array_equal(
            got_dx,
            saturate_signed(round_shift_signed(dy @ w.T, 2), 32),
        )
        np.testing.assert_array_equal(
            got_dw,
            saturate_signed(round_shift_signed(x.T @ dy, 1), 24),
        )
        np.testing.assert_array_equal(
            got_db,
            saturate_signed(round_shift_signed(dy.sum(axis=0), 1), 24),
        )

    def test_grouped_convolution_backward_and_fabric_col2im(self) -> None:
        x = self.rng.integers(-8, 9, size=(1, 4, 4, 5), dtype=np.int64)
        w = self.rng.integers(-20, 21, size=(4, 2, 3, 3), dtype=np.int64)
        bias = self.rng.integers(-30, 31, size=4, dtype=np.int64)
        dy = self.rng.integers(-12, 13, size=(1, 4, 2, 3), dtype=np.int64)
        program = compile_conv2d_backward(
            x,
            w,
            bias,
            dy,
            stride=2,
            padding=1,
            groups=2,
        )
        self.assertEqual(program.output_shape, (1, 4, 2, 3))
        self.assertEqual(program.partial_tile_count, 12)
        got_dx, got_dw, got_db = program.execute_batched(FullTrainingEmulator())

        expected_dx = np.zeros_like(x)
        expected_dw = np.zeros_like(w)
        expected_db = np.zeros_like(bias)
        for output_channel in range(4):
            group = output_channel // 2
            input_start = group * 2
            for output_y in range(2):
                for output_x in range(3):
                    gradient = int(dy[0, output_channel, output_y, output_x])
                    expected_db[output_channel] += gradient
                    for local_input in range(2):
                        for kernel_y in range(3):
                            input_y = output_y * 2 - 1 + kernel_y
                            for kernel_x in range(3):
                                input_x = output_x * 2 - 1 + kernel_x
                                if 0 <= input_y < 4 and 0 <= input_x < 5:
                                    expected_dw[output_channel, local_input, kernel_y, kernel_x] += (
                                        int(x[0, input_start + local_input, input_y, input_x])
                                        * gradient
                                    )
                                    expected_dx[0, input_start + local_input, input_y, input_x] += (
                                        int(w[output_channel, local_input, kernel_y, kernel_x])
                                        * gradient
                                    )
        np.testing.assert_array_equal(got_dx, expected_dx)
        np.testing.assert_array_equal(got_dw, expected_dw)
        np.testing.assert_array_equal(got_db, expected_db)

    def test_arbitrary_parameter_sgdw_and_int8_projection_run_in_fabric(self) -> None:
        weights = self.rng.integers(-500, 501, size=(3, 11), dtype=np.int64)
        gradients = self.rng.integers(-(1 << 20), 1 << 20, size=(3, 11), dtype=np.int64)
        velocities = self.rng.integers(-(1 << 20), 1 << 20, size=(3, 11), dtype=np.int64)
        expected_weight, expected_velocity = sgd_momentum_step(
            weights,
            gradients,
            velocities,
            655,
            58982,
            66,
        )
        emulator = FabricOperatorEmulator()
        got_weight, got_velocity = execute_sgdw_update(
            emulator,
            weights,
            gradients,
            velocities,
            learning_rate_q16=655,
            momentum_q16=58982,
            weight_decay_q16=66,
        )
        np.testing.assert_array_equal(got_weight, expected_weight)
        np.testing.assert_array_equal(got_velocity, expected_velocity)
        projected = execute_int8_weight_projection(emulator, got_weight)
        np.testing.assert_array_equal(projected, np.clip(got_weight, -128, 127).astype(np.int8))
        self.assertEqual(emulator.command_count, 10)

    def test_rejects_misaligned_or_out_of_range_operands(self) -> None:
        with self.assertRaises(ValueError):
            compile_linear_backward(np.zeros((2, 3)), np.zeros((4, 5)), np.zeros((2, 5)))
        with self.assertRaises(ValueError):
            compile_linear_backward(np.array([[32768]]), np.array([[1]]), np.array([[1]]))
        with self.assertRaises(ValueError):
            compile_affine_backward(
                np.zeros((1, 1), dtype=np.int64),
                np.zeros((1, 2), dtype=np.int64),
                np.zeros(3, dtype=np.int64),
                np.zeros((1, 2), dtype=np.int64),
            )

    def test_lif_bptt_fabric_schedule_matches_fixed_point_recurrence(self) -> None:
        time_steps = 4
        neurons = 19
        threshold = 1 << 10
        membrane = self.rng.integers(
            threshold - 4096,
            threshold + 4097,
            size=(time_steps, neurons),
            dtype=np.int64,
        )
        spikes = (membrane >= threshold).astype(np.int64)
        grad_spikes = self.rng.integers(
            -(1 << 20), 1 << 20, size=(time_steps, neurons), dtype=np.int64
        )
        grad_membrane = self.rng.integers(
            -(1 << 18), 1 << 18, size=(time_steps, neurons), dtype=np.int64
        )
        table, address_shift = sigmoid_surrogate_lut_q15()
        self.assertEqual(int(table[128]), 32767)
        addresses = surrogate_lut_addresses(membrane, threshold, address_shift)
        self.assertTrue(np.all(addresses >= 0))
        self.assertTrue(np.all(addresses <= 255))
        expected_input, expected_h = lif_bptt_fixed(
            membrane,
            spikes,
            grad_spikes,
            threshold=threshold,
            membrane_gradients=grad_membrane,
            surrogate_table_q15=table,
            surrogate_address_shift=address_shift,
        )
        emulator = FabricOperatorEmulator()
        got_input, got_h = execute_lif_bptt(
            emulator,
            membrane,
            spikes,
            grad_spikes,
            threshold=threshold,
            membrane_gradients=grad_membrane,
            surrogate_table_q15=table,
            surrogate_address_shift=address_shift,
        )
        np.testing.assert_array_equal(got_input, expected_input)
        np.testing.assert_array_equal(got_h, expected_h)
        self.assertEqual(emulator.command_count, 1 + time_steps * 3 * 7)

    def test_maxpool_backward_matches_first_winner_and_overlap_accumulation(self) -> None:
        values = self.rng.integers(-20, 21, size=(1, 2, 5, 6), dtype=np.int64)
        values[:, :, 1:4, 1:5] = 7
        values[0, 0, 0, 0] = -(1 << 31)
        gradients = self.rng.integers(-1000, 1001, size=(1, 2, 3, 3), dtype=np.int64)
        expected_gradient, expected_winners = maxpool2d_backward_fixed(
            values,
            gradients,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        emulator = FabricOperatorEmulator()
        got_gradient, got_winners = execute_maxpool2d_backward(
            emulator,
            values,
            gradients,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        np.testing.assert_array_equal(got_winners, expected_winners)
        np.testing.assert_array_equal(got_gradient, expected_gradient)
        self.assertGreater(emulator.command_count, 100)

    def test_real_spikingformer_manifest_has_no_training_coverage_holes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = build_spikingformer_training_coverage(
            root / "outputs" / "spikingformer_reference" / "spikingformer_execution_manifest.json",
            root / "outputs" / "spikingformer_hardware_parameters" / "spikingformer_neuraedge_weights.json",
        )
        summary = result["summary"]
        self.assertEqual(summary["execution_operator_count"], 43)
        self.assertEqual(summary["covered_execution_operator_count"], 43)
        self.assertEqual(summary["learned_affine_group_count"], 30)
        self.assertEqual(summary["backbone_projection_count"], 29)
        self.assertEqual(summary["dynamic_attention_matmul_count"], 8)
        self.assertEqual(summary["lif_node_count"], 32)
        self.assertEqual(summary["trainable_deployed_parameter_elements"], 9_295_978)
        self.assertEqual(summary["uncovered_operator_count"], 0)
        self.assertTrue(summary["full_deployed_backbone_covered"])
        self.assertEqual(
            {item["name"] for item in result["operator_coverage"]},
            {item["name"] for item in result["learned_affine_coverage"]}
            | {item["name"] for item in result["dynamic_attention_coverage"]}
            | {f"block.{index}.residuals" for index in range(4)}
            | {"global_average_pool"},
        )


if __name__ == "__main__":
    unittest.main()
