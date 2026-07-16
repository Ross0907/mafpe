from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neuraedge_golden import VectorOperation, round_shift_signed, vector_alu
from neuraedge_protocol import Opcode, ProgramRecord, unpack_vector_result
from spikingformer_accelerator import (
    SpikingformerProjectionExecutor,
    binary_max_pool2d_on_accelerator,
    global_spatiotemporal_sum_on_accelerator,
    lif_sequence_on_accelerator,
    lif_sequence_reference,
    lif_sequence_with_trace_on_accelerator,
    lif_sequence_with_trace_reference,
    power_of_two_requantize_on_accelerator,
    requantize_projection_on_accelerator,
    requantize_projection_reference,
    vector_binary_on_accelerator,
)
from spikingformer_parameters import ProjectionParameters, SpikingformerParameterArchive
from tests.test_neuraedge_compiler import TensorTransactionEmulator


def _signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


class VectorProgramEmulator:
    def execute_program(self, records, *, timeout_seconds: float = 30.0):
        del timeout_seconds
        output: list[int] = []
        for record in records:
            self.assert_vector_record(record)
            command = record.command.value
            operation = (command >> 4) & 0xF
            multiplier = _signed(command >> 8, 16)
            shift = _signed(command >> 24, 5) if operation == 9 else (command >> 24) & 0x1F
            offset = _signed(command >> 29, 16)
            clamp_word = record.input_beats[0].data
            clamp_minimum = _signed(clamp_word, 32)
            clamp_maximum = _signed(clamp_word >> 32, 32)
            a = unpack_vector_result([beat.data for beat in record.input_beats[1:5]])
            b = unpack_vector_result([beat.data for beat in record.input_beats[5:9]])
            lanes = vector_alu(
                VectorOperation(operation), np.asarray(a), np.asarray(b),
                multiplier_q15=multiplier, right_shift=shift,
                output_zero_point=offset, clamp_minimum=clamp_minimum,
                clamp_maximum=clamp_maximum)
            for lane in range(0, 8, 2):
                output.append(
                    (int(lanes[lane]) & 0xFFFFFFFF)
                    | ((int(lanes[lane + 1]) & 0xFFFFFFFF) << 32))
        return tuple(output)

    @staticmethod
    def assert_vector_record(record: ProgramRecord) -> None:
        if len(record.input_beats) != 9 or record.output_beats != 4:
            raise AssertionError("invalid vector record")


class CombinedProgramEmulator:
    def __init__(self) -> None:
        self.tensor = TensorTransactionEmulator()
        self.vector = VectorProgramEmulator()

    def execute_program(self, records, *, timeout_seconds: float = 30.0):
        if records and all(record.command.opcode == Opcode.VECTOR_ALU for record in records):
            return self.vector.execute_program(records, timeout_seconds=timeout_seconds)
        return self.tensor.execute_program(records, timeout_seconds=timeout_seconds)


class OneProjectionArchive:
    def __init__(self, parameters: ProjectionParameters) -> None:
        self.parameters = parameters

    def projection(self, name: str) -> ProjectionParameters:
        if name != self.parameters.name:
            raise KeyError(name)
        return self.parameters


class LifProgramEmulator:
    def __init__(self) -> None:
        self.membrane = np.zeros(32, dtype=np.int64)
        self.timesteps = 0

    def execute_program(self, records, *, timeout_seconds: float = 30.0):
        del timeout_seconds
        output: list[int] = []
        for record in records:
            command = record.command.value
            if record.command.opcode != Opcode.LIF_STEP:
                raise AssertionError("expected an LIF record")
            leak_shift = (command >> 4) & 0x1F
            threshold = _signed(command >> 9, 24)
            reset = _signed(command >> 33, 24)
            if (command >> 57) & 1:
                self.membrane.fill(reset)
                self.timesteps = 0
            currents: list[int] = []
            for beat in record.input_beats:
                currents.extend(_signed(beat.data >> (16 * lane), 16) for lane in range(4))
            self.membrane = np.clip(
                self.membrane - (self.membrane >> leak_shift) + np.asarray(currents),
                -(1 << 23),
                (1 << 23) - 1,
            )
            pre_reset_membrane = self.membrane.copy()
            fired = self.membrane >= threshold
            self.membrane = np.where(fired, reset, self.membrane)
            self.timesteps += 1
            spike_word = sum(int(fired[lane]) << lane for lane in range(32))
            output.append(spike_word | (self.timesteps << 32))
            if (command >> 58) & 1:
                packed = sum(
                    (int(pre_reset_membrane[lane]) & 0xFFFFFF) << (24 * lane)
                    for lane in range(32)
                )
                output.extend(
                    (packed >> (64 * word)) & 0xFFFFFFFFFFFFFFFF
                    for word in range(12)
                )
                if record.output_beats != 13:
                    raise AssertionError("traced LIF record must request thirteen beats")
            elif record.output_beats != 1:
                raise AssertionError("untraced LIF record must request one beat")
        return tuple(output)


class SpikingformerAcceleratorTests(unittest.TestCase):
    @staticmethod
    def parameters(name: str, weight: np.ndarray, opcode: str) -> ProjectionParameters:
        channels = weight.shape[1]
        return ProjectionParameters(
            name,
            weight.astype(np.int8),
            np.arange(channels, dtype=np.int32) * 17 - 31,
            np.asarray([16384 + 733 * channel for channel in range(channels)], dtype=np.int16),
            np.asarray([(channel % 3) - 1 for channel in range(channels)], dtype=np.int8),
            opcode,
            (channels, weight.shape[0], 1),
            10,
        )

    def test_fabric_projection_requantizer_matches_independent_reference(self) -> None:
        rng = np.random.default_rng(20260713)
        rows = 19
        channels = 13
        parameters = ProjectionParameters(
            "test",
            np.zeros((17, channels), dtype=np.int8),
            rng.integers(-3000, 3001, size=channels, dtype=np.int32),
            rng.integers(-32768, 32768, size=channels, dtype=np.int16),
            rng.integers(-10, 11, size=channels, dtype=np.int8),
            "COMPUTE_BINARY",
            (channels, 17, 1, 1),
            10,
        )
        accumulators = rng.integers(-500_000, 500_001, size=(rows, channels), dtype=np.int64)
        expected = requantize_projection_reference(accumulators, parameters)
        actual = requantize_projection_on_accelerator(
            VectorProgramEmulator(), accumulators, parameters)
        np.testing.assert_array_equal(actual, expected)

    def test_binary_projection_executes_tensor_and_vector_programs(self) -> None:
        rng = np.random.default_rng(11)
        spikes = rng.integers(0, 2, size=(11, 7), dtype=np.uint8)
        weight = rng.integers(-127, 128, size=(7, 5), dtype=np.int16).astype(np.int8)
        parameters = self.parameters("binary", weight, "COMPUTE_BINARY")
        executor = SpikingformerProjectionExecutor(
            OneProjectionArchive(parameters), CombinedProgramEmulator())
        actual = executor.matrix("binary", spikes)
        expected_accumulator = spikes.astype(np.int64) @ weight.astype(np.int64)
        expected = requantize_projection_reference(expected_accumulator, parameters)
        np.testing.assert_array_equal(actual, expected)

    def test_asymmetric_dense_projection_executes_exact_zero_point_math(self) -> None:
        rng = np.random.default_rng(12)
        activations = rng.integers(0, 256, size=(9, 13), dtype=np.uint8)
        weight = rng.integers(-127, 128, size=(13, 6), dtype=np.int16).astype(np.int8)
        parameters = self.parameters("dense", weight, "COMPUTE_DENSE_OUTER")
        executor = SpikingformerProjectionExecutor(
            OneProjectionArchive(parameters), CombinedProgramEmulator())
        actual = executor.matrix("dense", activations, activation_zero_point=123)
        expected_accumulator = (
            activations.astype(np.int64) - 123) @ weight.astype(np.int64)
        expected = requantize_projection_reference(expected_accumulator, parameters)
        np.testing.assert_array_equal(actual, expected)

    def test_chunked_lif_scheduler_preserves_every_logical_membrane(self) -> None:
        rng = np.random.default_rng(13)
        values = rng.integers(-5000, 5001, size=(4, 3, 5, 7), dtype=np.int64)
        expected = lif_sequence_reference(
            values, threshold_q10=512, leak_shift=1, reset_q10=0)
        actual = lif_sequence_on_accelerator(
            LifProgramEmulator(), values,
            threshold_q10=512, leak_shift=1, reset_q10=0)
        np.testing.assert_array_equal(actual, expected)

    def test_traced_lif_scheduler_returns_exact_pre_reset_membrane(self) -> None:
        rng = np.random.default_rng(1301)
        values = rng.integers(-9000, 9001, size=(5, 2, 7, 5), dtype=np.int64)
        expected_spikes, expected_membrane = lif_sequence_with_trace_reference(
            values, threshold_q10=700, leak_shift=1, reset_q10=-17)
        actual_spikes, actual_membrane = lif_sequence_with_trace_on_accelerator(
            LifProgramEmulator(), values,
            threshold_q10=700, leak_shift=1, reset_q10=-17)
        np.testing.assert_array_equal(actual_spikes, expected_spikes)
        np.testing.assert_array_equal(actual_membrane, expected_membrane)
        self.assertTrue(np.any(actual_spikes))
        self.assertTrue(np.any(actual_membrane[actual_spikes.astype(bool)] >= 700))

    def test_real_block0_archive_executes_exact_exported_integer_equation(self) -> None:
        manifest = Path(__file__).resolve().parents[1] / (
            "outputs/spikingformer_hardware_parameters/"
            "spikingformer_neuraedge_weights.json")
        if not manifest.exists():
            self.skipTest("full trained parameter archive is not present")
        archive = SpikingformerParameterArchive(manifest)
        parameters = archive.projection("patch_embed.block0")
        rng = np.random.default_rng(14)
        activations = rng.integers(
            0, 256, size=(3, parameters.weight_k_by_n.shape[0]), dtype=np.uint8)
        executor = SpikingformerProjectionExecutor(archive, CombinedProgramEmulator())
        actual = executor.matrix(
            "patch_embed.block0", activations, activation_zero_point=123)
        accumulators = (
            activations.astype(np.int64) - 123
        ) @ parameters.weight_k_by_n.astype(np.int64)
        expected = requantize_projection_reference(accumulators, parameters)
        np.testing.assert_array_equal(actual, expected)

    def test_vector_residual_and_binary_max_pool_cover_edge_lanes(self) -> None:
        left = np.asarray([-(1 << 31), -3, 0, 7, (1 << 31) - 1, 4, 5, 6, 7])
        right = np.asarray([-1, 8, 0, -2, 1, 3, 2, 1, 0])
        added = vector_binary_on_accelerator(
            VectorProgramEmulator(), VectorOperation.ADD_SAT, left, right)
        np.testing.assert_array_equal(
            added, np.clip(left.astype(np.int64) + right, -(1 << 31), (1 << 31) - 1))

        rng = np.random.default_rng(15)
        spikes = rng.integers(0, 2, size=(2, 3, 5, 7), dtype=np.uint8)
        actual = binary_max_pool2d_on_accelerator(VectorProgramEmulator(), spikes)
        padded = np.pad(spikes, ((0, 0), (0, 0), (1, 1), (1, 1)))
        candidates = [
            padded[:, :, y:y + 5:2, x:x + 7:2]
            for y in range(3) for x in range(3)
        ]
        expected = np.maximum.reduce(candidates)
        np.testing.assert_array_equal(actual, expected)

    def test_global_pooling_and_head_division_execute_in_vector_fabric(self) -> None:
        values = np.arange(-120, 120, dtype=np.int64).reshape(3, 2, 5, 4, 2)
        pooled = global_spatiotemporal_sum_on_accelerator(
            VectorProgramEmulator(), values)
        np.testing.assert_array_equal(pooled, values.sum(axis=(0, 3, 4)))
        divided = power_of_two_requantize_on_accelerator(
            VectorProgramEmulator(),
            pooled,
            division_shift=5,
            clamp_minimum=-17,
            clamp_maximum=19,
        )
        np.testing.assert_array_equal(
            divided,
            np.clip(round_shift_signed(pooled, 5), -17, 19),
        )


if __name__ == "__main__":
    unittest.main()
