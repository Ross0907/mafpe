from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from spikingformer_reference import (
    MultiStepHardResetLIF,
    Spikingformer,
    fold_batch_norm,
)
from evaluate_spikingformer import CalibrationObserver
from spikingformer_quantization import (
    FixedPointHardResetLIF,
    fake_fixed_point,
    prepare_hardware_emulation,
    quantize_cifar_input_u8,
)


class MultiStepLIFTests(unittest.TestCase):
    def test_hard_reset_sequence(self) -> None:
        neuron = MultiStepHardResetLIF(tau=2.0, threshold=1.0)
        sequence = torch.tensor([[1.0], [1.0], [1.0], [2.0], [1.0]])
        expected = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0]])
        torch.testing.assert_close(neuron(sequence), expected)

    def test_fixed_point_lif_matches_rtl_step_order(self) -> None:
        neuron = FixedPointHardResetLIF(threshold=1.0, fractional_bits=4)
        sequence = torch.tensor([[1.0], [1.0], [1.0], [2.0], [1.0]])
        expected = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0]])
        torch.testing.assert_close(neuron(sequence), expected)

    def test_attention_threshold_is_inclusive(self) -> None:
        neuron = MultiStepHardResetLIF(tau=2.0, threshold=0.5)
        sequence = torch.tensor([[1.0], [0.0]])
        expected = torch.tensor([[1.0], [0.0]])
        torch.testing.assert_close(neuron(sequence), expected)


class BatchNormFoldingTests(unittest.TestCase):
    def test_conv2d_fold_matches_eval_graph(self) -> None:
        torch.manual_seed(7)
        convolution = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=False)
        batch_norm = nn.BatchNorm2d(5)
        batch_norm.running_mean.copy_(torch.randn(5))
        batch_norm.running_var.copy_(torch.rand(5) + 0.2)
        batch_norm.weight.data.copy_(torch.randn(5))
        batch_norm.bias.data.copy_(torch.randn(5))
        convolution.eval()
        batch_norm.eval()
        image = torch.randn(2, 3, 7, 7)

        folded_weight, folded_bias = fold_batch_norm(convolution, batch_norm)
        folded = nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True)
        folded.weight.data.copy_(folded_weight)
        folded.bias.data.copy_(folded_bias)
        folded.eval()

        torch.testing.assert_close(
            folded(image), batch_norm(convolution(image)), rtol=2e-5, atol=2e-6)

    def test_conv1d_fold_matches_eval_graph(self) -> None:
        torch.manual_seed(11)
        convolution = nn.Conv1d(4, 6, kernel_size=1, bias=True)
        batch_norm = nn.BatchNorm1d(6)
        batch_norm.running_mean.copy_(torch.randn(6))
        batch_norm.running_var.copy_(torch.rand(6) + 0.2)
        convolution.eval()
        batch_norm.eval()
        values = torch.randn(3, 4, 9)

        folded_weight, folded_bias = fold_batch_norm(convolution, batch_norm)
        folded = nn.Conv1d(4, 6, kernel_size=1, bias=True)
        folded.weight.data.copy_(folded_weight)
        folded.bias.data.copy_(folded_bias)
        folded.eval()

        torch.testing.assert_close(
            folded(values), batch_norm(convolution(values)), rtol=2e-5, atol=2e-6)


class ReferenceGraphTests(unittest.TestCase):
    def test_small_graph_shape_and_binary_observations(self) -> None:
        torch.manual_seed(13)
        model = Spikingformer(
            time_steps=2,
            input_channels=3,
            classes=10,
            embed_dim=24,
            heads=3,
            depth=1,
            mlp_ratio=2,
        ).eval()
        observed_spikes: list[torch.Tensor] = []

        def observer(_name: str, value: torch.Tensor, is_spike: bool) -> None:
            if is_spike:
                observed_spikes.append(value.detach())

        with torch.no_grad():
            result = model(torch.randn(2, 3, 8, 8), observer=observer)
        self.assertEqual(tuple(result.shape), (2, 10))
        self.assertGreater(len(observed_spikes), 0)
        for spike in observed_spikes:
            self.assertTrue(torch.all((spike == 0) | (spike == 1)).item())

    def test_calibration_observer_aggregates_without_feature_storage(self) -> None:
        observer = CalibrationObserver()
        observer("analog", torch.tensor([[-1.0, 1.0], [2.0, 0.0]]), False)
        observer("analog", torch.tensor([[3.0, 1.0]]), False)
        report = observer.as_dict()["analog"]
        self.assertEqual(report["elements"], 6)
        self.assertEqual(report["batches"], 2)
        self.assertEqual(report["minimum"], -1.0)
        self.assertEqual(report["maximum"], 3.0)
        self.assertAlmostEqual(report["mean"], 1.0)
        self.assertAlmostEqual(report["nonzero_fraction"], 5.0 / 6.0)

    def test_hardware_emulation_quantizes_all_learned_operators(self) -> None:
        torch.manual_seed(17)
        model = Spikingformer(
            time_steps=2,
            input_channels=3,
            classes=10,
            embed_dim=24,
            heads=3,
            depth=1,
            mlp_ratio=2,
        ).eval()
        quantized, report = prepare_hardware_emulation(
            model, fractional_bits=8, weight_granularity="tensor")
        self.assertEqual(len(report.quantized_weights), 12)
        self.assertTrue(all(item.integer_minimum >= -127 for item in report.quantized_weights))
        self.assertTrue(all(item.integer_maximum <= 127 for item in report.quantized_weights))
        image = torch.randn(2, 3, 8, 8)
        image_quantized, scale, zero_point = quantize_cifar_input_u8(image)
        self.assertGreater(scale, 0.0)
        self.assertTrue(0 <= zero_point <= 255)
        self.assertEqual(report.head_activation_scale, 0.125)
        self.assertGreater(report.head_accumulator_scale, 0.0)
        with torch.no_grad():
            result = quantized(image_quantized)
        self.assertEqual(tuple(result.shape), (2, 10))
        fixed = fake_fixed_point(torch.tensor([-1.1, 0.1, 1.1]), 4)
        torch.testing.assert_close(fixed * 16, torch.round(fixed * 16))


if __name__ == "__main__":
    unittest.main()
