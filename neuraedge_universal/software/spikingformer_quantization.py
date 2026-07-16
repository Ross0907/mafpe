"""Quantization functions that mirror the implemented NeuraEdge arithmetic."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn as nn

from neuraedge_golden import quantize_flexible_scale
from spikingformer_reference import MultiStepHardResetLIF, Spikingformer


WeightGranularity = Literal["tensor", "channel"]


@dataclass(frozen=True)
class QuantizedWeight:
    module: str
    shape: tuple[int, ...]
    granularity: WeightGranularity
    integer_minimum: int
    integer_maximum: int
    scale_minimum: float
    scale_maximum: float
    mean_absolute_error: float
    maximum_absolute_error: float


@dataclass(frozen=True)
class HardwareEmulationReport:
    fractional_bits: int
    fixed_point_scale: int
    input_scale: float
    input_zero_point: int
    head_activation_scale: float
    head_accumulator_scale: float
    weight_granularity: WeightGranularity
    quantized_weights: tuple[QuantizedWeight, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["quantized_weights"] = [asdict(item) for item in self.quantized_weights]
        return result


class FixedPointHardResetLIF(MultiStepHardResetLIF):
    """Bit-exact model of ``neuraedge_lif_bank`` for one logical vector.

    The RTL first leaks ``v`` by ``v - (v >>> 1)`` and then adds a signed
    16-bit current.  Supplying ``x >>> 1`` as that current implements the
    checkpoint equation ``v/2 + x/2`` with deterministic integer rounding.
    Membranes saturate to signed 24-bit values before the threshold test.
    """

    def __init__(self, *, threshold: float, fractional_bits: int) -> None:
        super().__init__(tau=2.0, threshold=threshold)
        if not 0 <= fractional_bits <= 14:
            raise ValueError("fractional_bits must keep halved current in signed 16 bits")
        self.fractional_bits = fractional_bits
        self.fixed_scale = 1 << fractional_bits
        self.threshold_integer = int(round(threshold * self.fixed_scale))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        input_integer = torch.round(sequence * self.fixed_scale).to(torch.int64)
        voltage = torch.zeros_like(input_integer[0])
        spikes: list[torch.Tensor] = []
        minimum_membrane = -(1 << 23)
        maximum_membrane = (1 << 23) - 1
        for timestep in range(input_integer.shape[0]):
            current = torch.bitwise_right_shift(input_integer[timestep], 1)
            current = current.clamp(-(1 << 15), (1 << 15) - 1)
            leaked = voltage - torch.bitwise_right_shift(voltage, 1)
            voltage = (leaked + current).clamp(minimum_membrane, maximum_membrane)
            spike = voltage >= self.threshold_integer
            voltage = torch.where(spike, torch.zeros_like(voltage), voltage)
            spikes.append(spike.to(sequence.dtype))
        return torch.stack(spikes, dim=0)


def quantize_cifar_input_u8(image: torch.Tensor) -> tuple[torch.Tensor, float, int]:
    """Quantize the complete legal normalized CIFAR-10 input domain to uint8."""

    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=image.device)
    standard_deviation = torch.tensor((0.2470, 0.2435, 0.2616), device=image.device)
    minimum = float(torch.min(-mean / standard_deviation).cpu())
    maximum = float(torch.max((1.0 - mean) / standard_deviation).cpu())
    scale = (maximum - minimum) / 255.0
    zero_point = int(round(-minimum / scale))
    integer = torch.round(image / scale + zero_point).clamp(0, 255)
    return (integer - zero_point) * scale, scale, zero_point


def fake_fixed_point(value: torch.Tensor, fractional_bits: int) -> torch.Tensor:
    """Round and saturate an analog activation to signed 24-bit fixed point."""

    scale = 1 << fractional_bits
    integer = torch.round(value * scale).clamp(-(1 << 23), (1 << 23) - 1)
    return integer / scale


def prepare_hardware_emulation(
    model: Spikingformer,
    *,
    fractional_bits: int = 10,
    weight_granularity: WeightGranularity = "tensor",
    head_activation_scale: float = 0.125,
) -> tuple[Spikingformer, HardwareEmulationReport]:
    """Clone a trained model and apply the arithmetic supported by the RTL.

    Conv/linear weights become signed int8 and are immediately dequantized for
    PyTorch execution.  This is algebraically equivalent to an int32 hardware
    accumulator followed by its recorded scale.  BatchNorm outputs are rounded
    to the common signed fixed-point format used by residuals and LIF neurons.
    """

    if weight_granularity not in {"tensor", "channel"}:
        raise ValueError("weight_granularity must be tensor or channel")
    if not 0 <= fractional_bits <= 14:
        raise ValueError("fractional_bits must be in [0, 14]")
    if head_activation_scale <= 0.0:
        raise ValueError("head_activation_scale must be positive")
    quantized_model = copy.deepcopy(model).eval()
    weight_reports: list[QuantizedWeight] = []
    weight_scales: dict[str, torch.Tensor] = {}

    for name, module in quantized_model.named_modules():
        if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            continue
        weight = module.weight.detach()
        if weight_granularity == "channel":
            dimensions = tuple(range(1, weight.ndim))
            maximum = weight.abs().amax(dim=dimensions, keepdim=True)
        else:
            maximum = weight.abs().amax().reshape((1,) * weight.ndim)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        integer = torch.round(weight / scale).clamp(-127, 127)
        dequantized = integer * scale
        error = (dequantized - weight).abs()
        module.weight.data.copy_(dequantized)
        weight_scales[name] = scale.detach().reshape(-1)
        weight_reports.append(QuantizedWeight(
            module=name,
            shape=tuple(weight.shape),
            granularity=weight_granularity,
            integer_minimum=int(integer.amin().item()),
            integer_maximum=int(integer.amax().item()),
            scale_minimum=float(scale.amin().item()),
            scale_maximum=float(scale.amax().item()),
            mean_absolute_error=float(error.mean().item()),
            maximum_absolute_error=float(error.amax().item()),
        ))

    _replace_lif_modules(quantized_model, fractional_bits)
    _install_projection_requantizers(
        quantized_model,
        weight_scales,
        fractional_bits=fractional_bits,
        image_input_scale=input_scale_from_domain(),
    )
    quantized_model.head.register_forward_pre_hook(
        lambda _module, inputs, scale=head_activation_scale: (
            torch.round(inputs[0] / scale).clamp(-128, 127) * scale,
        ))
    head_weight_scale = float(quantized_model.head.weight.detach().abs().amax().item()) / 127.0
    head_accumulator_scale = head_activation_scale * head_weight_scale
    if quantized_model.head.bias is not None:
        quantized_model.head.bias.data.copy_(
            torch.round(quantized_model.head.bias.detach() / head_accumulator_scale)
            * head_accumulator_scale)

    input_probe = torch.zeros(1, 3, 1, 1)
    _, input_scale, input_zero_point = quantize_cifar_input_u8(input_probe)
    report = HardwareEmulationReport(
        fractional_bits=fractional_bits,
        fixed_point_scale=1 << fractional_bits,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        head_activation_scale=head_activation_scale,
        head_accumulator_scale=head_accumulator_scale,
        weight_granularity=weight_granularity,
        quantized_weights=tuple(weight_reports),
    )
    return quantized_model, report


def _replace_lif_modules(module: nn.Module, fractional_bits: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, MultiStepHardResetLIF):
            setattr(module, name, FixedPointHardResetLIF(
                threshold=child.threshold,
                fractional_bits=fractional_bits,
            ))
        else:
            _replace_lif_modules(child, fractional_bits)


def input_scale_from_domain() -> float:
    probe = torch.zeros(1, 3, 1, 1)
    _, scale, _ = quantize_cifar_input_u8(probe)
    return scale


def _install_projection_requantizers(
    model: Spikingformer,
    weight_scales: dict[str, torch.Tensor],
    *,
    fractional_bits: int,
    image_input_scale: float,
) -> None:
    modules = dict(model.named_modules())
    output_scale = 1 << fractional_bits
    for batch_norm_name, batch_norm in modules.items():
        if not isinstance(batch_norm, (nn.BatchNorm1d, nn.BatchNorm2d)):
            continue
        if not batch_norm_name.endswith("_bn"):
            raise ValueError(f"cannot locate projection for {batch_norm_name}")
        convolution_name = batch_norm_name[:-3] + "_conv"
        convolution = modules.get(convolution_name)
        if not isinstance(convolution, (nn.Conv1d, nn.Conv2d)):
            raise ValueError(f"cannot locate {convolution_name}")
        raw_weight_scale = weight_scales[convolution_name]
        if raw_weight_scale.numel() == 1:
            weight_scale = raw_weight_scale.repeat(convolution.out_channels)
        else:
            weight_scale = raw_weight_scale
        projection_input_scale = (
            image_input_scale if convolution_name == "patch_embed.block0_conv" else 1.0)
        bn_scale = batch_norm.weight.detach() / torch.sqrt(
            batch_norm.running_var.detach() + batch_norm.eps)
        factor = projection_input_scale * weight_scale * bn_scale * output_scale
        multiplier_and_shift = [quantize_flexible_scale(float(item)) for item in factor]
        multiplier = torch.tensor(
            [item[0] for item in multiplier_and_shift], dtype=torch.int64)
        effective_shift = torch.tensor(
            [15 + item[1] for item in multiplier_and_shift], dtype=torch.int64)
        if convolution.bias is None:
            convolution_bias = torch.zeros_like(batch_norm.running_mean)
        else:
            convolution_bias = convolution.bias.detach()
        folded_bias = (
            (convolution_bias - batch_norm.running_mean.detach()) * bn_scale
            + batch_norm.bias.detach())
        bias_integer = torch.round(folded_bias * output_scale).to(torch.int64)
        batch_norm.register_forward_hook(_projection_hook(
            convolution_bias=convolution_bias,
            projection_input_scale=projection_input_scale,
            weight_scale=weight_scale,
            multiplier=multiplier,
            effective_shift=effective_shift,
            bias_integer=bias_integer,
            output_scale=output_scale,
        ))


def _projection_hook(
    *,
    convolution_bias: torch.Tensor,
    projection_input_scale: float,
    weight_scale: torch.Tensor,
    multiplier: torch.Tensor,
    effective_shift: torch.Tensor,
    bias_integer: torch.Tensor,
    output_scale: int,
):
    def requantize(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> torch.Tensor:
        convolution_output = inputs[0]
        channel_shape = (1, -1) + (1,) * (convolution_output.ndim - 2)
        device = convolution_output.device
        bias = convolution_bias.to(device).reshape(channel_shape)
        accumulator_scale = (
            weight_scale.to(device).reshape(channel_shape) * projection_input_scale)
        accumulator = torch.round(
            (convolution_output - bias) / accumulator_scale).to(torch.int64)
        q15 = multiplier.to(device).reshape(channel_shape)
        shifts = effective_shift.to(device).reshape(channel_shape)
        product = accumulator * q15
        magnitude = product.abs()
        rounding_bias = torch.where(
            shifts == 0,
            torch.zeros_like(shifts),
            torch.bitwise_left_shift(torch.ones_like(shifts), shifts - 1),
        )
        rounded_magnitude = torch.bitwise_right_shift(
            magnitude + rounding_bias, shifts)
        rounded = torch.where(product < 0, -rounded_magnitude, rounded_magnitude)
        quantized = rounded + bias_integer.to(device).reshape(channel_shape)
        quantized = quantized.clamp(-(1 << 23), (1 << 23) - 1)
        return quantized.to(convolution_output.dtype) / output_scale

    return requantize
