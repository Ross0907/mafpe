from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuraedge_golden import round_shift_signed
from spikingformer_parameters import HeadParameters, ProjectionParameters
from spikingformer_training_runtime import (
    BlockGradient,
    MasterTensor,
    ParameterGradient,
    TrainableSpikingformerState,
    SpikingformerOnDeviceTrainer,
    add_block_gradients,
    backward_requantize_projection,
    execute_affine_backward_block,
    execute_cross_entropy_gradient_q15,
    execute_power_of_two_scale,
    execute_softmax_q15,
    normalize_gradient_s16,
)
from tests.test_neuraedge_training import FullTrainingEmulator


class TinyParameterArchive:
    def __init__(self) -> None:
        self.manifest = {
            "binary_sha256": "a" * 64,
            "fixed_point": {
                "activation_fractional_bits": 10,
                "image_input_scale": 0.02,
                "image_input_zero_point": 123,
                "head_activation_scale": 0.125,
            },
        }
        self._projection = ProjectionParameters(
            "tiny",
            np.asarray([[3, -4], [-2, 5]], dtype=np.int8),
            np.asarray([7, -9], dtype=np.int32),
            np.asarray([16384, 24576], dtype=np.int16),
            np.asarray([-1, 1], dtype=np.int8),
            "COMPUTE_BINARY",
            (2, 2, 1),
            10,
        )
        self._head = HeadParameters(
            np.asarray([[6, -7], [8, 9]], dtype=np.int8),
            np.asarray([11, -13], dtype=np.int32),
        )

    @property
    def projection_names(self) -> tuple[str, ...]:
        return ("tiny",)

    def projection(self, name: str) -> ProjectionParameters:
        if name != "tiny":
            raise KeyError(name)
        return self._projection

    @property
    def head(self) -> HeadParameters:
        return self._head


class TinyFullGraphArchive:
    def __init__(self) -> None:
        self.manifest = {
            "binary_sha256": "b" * 64,
            "fixed_point": {
                "activation_fractional_bits": 10,
                "image_input_scale": 0.02,
                "image_input_zero_point": 123,
                "head_activation_scale": 0.125,
            },
            "head": {"accumulator_scale": 1.0 / 1024.0},
        }
        self._projections: dict[str, ProjectionParameters] = {}

        def add_conv(name: str, input_channels: int, output_channels: int, dense=False) -> None:
            reduction = input_channels * 3 * 3
            weight = np.fromfunction(
                lambda row, column: ((row + 2 * column) % 3) - 1,
                (reduction, output_channels),
                dtype=int,
            ).astype(np.int8)
            self._projections[name] = ProjectionParameters(
                name,
                weight,
                np.full(output_channels, 2200, dtype=np.int32),
                np.full(output_channels, 16384, dtype=np.int16),
                np.full(output_channels, -1, dtype=np.int8),
                "COMPUTE_DENSE_OUTER" if dense else "COMPUTE_BINARY",
                (output_channels, input_channels, 3, 3),
                10,
            )

        add_conv("patch_embed.block0", 3, 2, dense=True)
        for index in range(1, 5):
            add_conv(f"patch_embed.block{index}", 2, 2)

        def add_matrix(name: str, inputs: int, outputs: int, bias: int = 2200) -> None:
            weight = np.fromfunction(
                lambda row, column: ((2 * row + column) % 3) - 1,
                (inputs, outputs),
                dtype=int,
            ).astype(np.int8)
            self._projections[name] = ProjectionParameters(
                name,
                weight,
                np.full(outputs, bias, dtype=np.int32),
                np.full(outputs, 16384, dtype=np.int16),
                np.full(outputs, -1, dtype=np.int8),
                "COMPUTE_BINARY",
                (outputs, inputs, 1),
                10,
            )

        for name in ("q", "k", "v"):
            add_matrix(f"block.0.attn.{name}", 2, 2)
        add_matrix("block.0.attn.proj", 2, 2, bias=0)
        add_matrix("block.0.mlp.fc1", 2, 3)
        add_matrix("block.0.mlp.fc2", 3, 2, bias=0)
        self._head = HeadParameters(
            np.asarray([[3, -2, 1], [-1, 2, 4]], dtype=np.int8),
            np.asarray([17, -11, 5], dtype=np.int32),
        )

    @property
    def projection_names(self) -> tuple[str, ...]:
        return tuple(self._projections)

    def projection(self, name: str) -> ProjectionParameters:
        return self._projections[name]

    @property
    def head(self) -> HeadParameters:
        return self._head


class SpikingformerTrainingRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.driver = FullTrainingEmulator()

    def test_block_floating_shift_normalize_and_add_are_exact(self) -> None:
        values = np.asarray([[-2_000_000, -65537, 0, 65537, 2_000_000]], dtype=np.int64)
        shifted = execute_power_of_two_scale(self.driver, values, -4)
        np.testing.assert_array_equal(shifted, round_shift_signed(values, 4))
        normalized = normalize_gradient_s16(self.driver, BlockGradient(values, -15))
        self.assertGreater(normalized.exponent, -15)
        self.assertLessEqual(int(np.abs(normalized.values).max()), 32767)
        left = BlockGradient(np.asarray([100, -200, 300]), -12)
        right = BlockGradient(np.asarray([800, 400, -1200]), -15)
        added = add_block_gradients(self.driver, left, right)
        self.assertEqual(added.exponent, -12)
        np.testing.assert_array_equal(
            added.values,
            left.values + round_shift_signed(right.values, 3),
        )

    def test_projection_derivative_uses_exported_channel_scales(self) -> None:
        parameters = TinyParameterArchive().projection("tiny")
        gradient = BlockGradient(
            np.asarray([[31000, -27000], [-11000, 9000], [777, -333]], dtype=np.int64),
            -15,
        )
        actual = backward_requantize_projection(self.driver, gradient, parameters)
        boost = gradient.exponent - actual.exponent
        expected = np.empty_like(gradient.values)
        for channel in range(2):
            expected[:, channel] = round_shift_signed(
                gradient.values[:, channel] * int(parameters.multiplier_q15[channel]),
                15 + int(parameters.binary_point_shift[channel]) - boost,
            )
        np.testing.assert_array_equal(actual.values, expected)
        self.assertLessEqual(int(np.abs(actual.values).max()), 32767)

    def test_affine_backward_returns_exact_dx_dw_and_bias(self) -> None:
        x = np.asarray([[3, -2, 1], [-4, 5, 2], [7, 1, -3]], dtype=np.int64)
        w = np.asarray([[2, -1], [4, 3], [-5, 6]], dtype=np.int64)
        bias = np.asarray([9, -8], dtype=np.int64)
        dy = np.asarray([[11, -7], [5, 13], [-3, 2]], dtype=np.int64)
        dx, parameter = execute_affine_backward_block(
            self.driver, x, w, bias, BlockGradient(dy, -15))
        self.assertEqual(dx.exponent, -15)
        self.assertEqual(parameter.weight.exponent, -15)
        np.testing.assert_array_equal(dx.values, dy @ w.T)
        np.testing.assert_array_equal(parameter.weight.values, x.T @ dy)
        np.testing.assert_array_equal(parameter.bias.values, dy.sum(axis=0))

    def test_hardware_softmax_and_batch_mean_cross_entropy_gradient(self) -> None:
        logits = np.asarray([[1200, -300, 77], [-80, 900, 100]], dtype=np.int64)
        probabilities = execute_softmax_q15(
            self.driver, logits, difference_shift=8)
        gradient = execute_cross_entropy_gradient_q15(
            self.driver, probabilities, np.asarray([0, 2]))
        self.assertEqual(gradient.exponent, -15)
        expected = probabilities.copy()
        expected[0, 0] -= 1 << 15
        expected[1, 2] -= 1 << 15
        expected = round_shift_signed(expected, 1)
        np.testing.assert_array_equal(gradient.values, expected)

    def test_low_rate_bias_update_retains_q16_learning_rate_resolution(self) -> None:
        tensor = MasterTensor(
            np.asarray([0], dtype=np.int16),
            np.asarray([0], dtype=np.int32),
            2,
            -(1 << 15),
            (1 << 15) - 1,
        )
        coefficients = TrainableSpikingformerState._optimizer_coefficients(
            tensor, 0.01, 0.9, 0.0)
        self.assertEqual(coefficients, (655, 58982, 0, 29))
        with self.assertRaises(ValueError):
            TrainableSpikingformerState._optimizer_coefficients(
                tensor, 1.0e-7, 0.9, 0.0)

    def test_full_parameter_update_and_atomic_checkpoint_round_trip(self) -> None:
        state = TrainableSpikingformerState(TinyParameterArchive())
        projection = state.projection("tiny")
        head = state.head
        gradients = {
            "tiny": ParameterGradient(
                BlockGradient(np.full(projection.weight_k_by_n.shape, 65536), -15),
                BlockGradient(np.full(projection.bias_q10.shape, -32768), -15),
            ),
            "head": ParameterGradient(
                BlockGradient(np.full(head.weight_k_by_n.shape, -65536), -15),
                BlockGradient(np.full(head.bias_accumulator_units.shape, 32768), -15),
            ),
        }
        state.apply_gradients(
            self.driver,
            gradients,
            learning_rate=0.25,
            momentum=0.0,
            weight_decay=0.0,
        )
        self.assertEqual(state.optimizer_step, 1)
        self.assertFalse(np.array_equal(state.projection("tiny").weight_k_by_n,
                                        projection.weight_k_by_n))
        self.assertFalse(np.array_equal(state.head.weight_k_by_n, head.weight_k_by_n))

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "latest_checkpoint.npz"
            digest = state.save_checkpoint(
                checkpoint,
                epoch=3,
                batch_index=17,
                best_accuracy_percent=95.75,
                extra={"reason": "unit-test"},
            )
            self.assertEqual(len(digest), 64)
            restored = TrainableSpikingformerState(TinyParameterArchive())
            metadata = restored.load_checkpoint(checkpoint)
            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["batch_index"], 17)
            self.assertEqual(restored.optimizer_step, 1)
            np.testing.assert_array_equal(
                restored.projection("tiny").weight_k_by_n,
                state.projection("tiny").weight_k_by_n,
            )
            np.testing.assert_array_equal(restored.head.weight_k_by_n, state.head.weight_k_by_n)

    def test_complete_spikingformer_topology_forward_backward_and_update(self) -> None:
        state = TrainableSpikingformerState(TinyFullGraphArchive())
        trainer = SpikingformerOnDeviceTrainer(
            state,
            self.driver,
            transformer_blocks=1,
            attention_heads=1,
            timesteps=2,
            input_shape=(3, 8, 8),
        )
        self.assertEqual(trainer.attention_scale_shift, 7)
        image = np.zeros((1, 3, 8, 8), dtype=np.float32)
        cache = trainer.forward(image)
        self.assertEqual(cache.final_value.shape, (2, 1, 2, 2, 2))
        self.assertEqual(len(cache.tokenizer), 5)
        self.assertEqual(len(cache.blocks), 1)
        self.assertEqual(cache.logits_accumulator.shape, (1, 3))
        backward = trainer.backward(cache, np.asarray([1]))
        self.assertTrue(np.isfinite(backward.loss))
        self.assertEqual(backward.input_gradient.values.shape, image.shape)
        self.assertEqual(
            set(backward.parameter_gradients),
            set(state.projection_names) | {"head"},
        )
        nonzero_gradients = 0
        for name, gradient in backward.parameter_gradients.items():
            expected_weight = state.head.weight_k_by_n.shape if name == "head" else (
                state.projection(name).weight_k_by_n.shape)
            expected_bias = state.head.bias_accumulator_units.shape if name == "head" else (
                state.projection(name).bias_q10.shape)
            self.assertEqual(gradient.weight.values.shape, expected_weight)
            self.assertEqual(gradient.bias.values.shape, expected_bias)
            nonzero_gradients += int(np.count_nonzero(gradient.weight.values))
            nonzero_gradients += int(np.count_nonzero(gradient.bias.values))
        self.assertGreater(nonzero_gradients, 0)
        state.apply_gradients(
            self.driver,
            backward.parameter_gradients,
            learning_rate=0.25,
            momentum=0.0,
            weight_decay=0.0,
        )
        self.assertEqual(state.optimizer_step, 1)


if __name__ == "__main__":
    unittest.main()
