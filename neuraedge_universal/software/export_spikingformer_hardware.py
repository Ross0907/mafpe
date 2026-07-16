"""Export the trained Spikingformer into the NeuraEdge binary parameter ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neuraedge_golden import quantize_flexible_scale
from spikingformer_quantization import quantize_cifar_input_u8
from spikingformer_reference import Spikingformer, load_checkpoint


ALIGNMENT = 64


@dataclass(frozen=True)
class BinarySection:
    name: str
    offset: int
    length: int
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class ProjectionExport:
    name: str
    source_weight_shape: tuple[int, ...]
    hardware_matrix_shape: tuple[int, int]
    input_encoding: str
    compute_opcode: str
    output_fractional_bits: int
    weight_scale: float
    accumulator_factor_minimum: float
    accumulator_factor_maximum: float
    represented_factor_maximum_error: float
    folded_bias_minimum: int
    folded_bias_maximum: int
    sections: tuple[BinarySection, ...]


class AlignedBinaryWriter:
    """Append typed arrays while retaining deterministic aligned offsets."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.sections: list[BinarySection] = []

    def append(self, name: str, value: np.ndarray) -> BinarySection:
        padding = (-len(self.data)) % ALIGNMENT
        self.data.extend(b"\x00" * padding)
        contiguous = np.ascontiguousarray(value)
        payload = contiguous.tobytes(order="C")
        section = BinarySection(
            name=name,
            offset=len(self.data),
            length=len(payload),
            dtype=contiguous.dtype.str,
            shape=tuple(contiguous.shape),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        self.data.extend(payload)
        self.sections.append(section)
        return section


def quantize_weight_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric signed-int8 quantization used by the measured 95.4% model."""

    maximum = float(weight.detach().abs().amax().item())
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    integer = torch.round(weight.detach() / scale).clamp(-127, 127).to(torch.int8)
    return integer, scale


def export_projection(
    writer: AlignedBinaryWriter,
    *,
    name: str,
    convolution: nn.Conv1d | nn.Conv2d,
    batch_norm: nn.BatchNorm1d | nn.BatchNorm2d,
    input_scale: float,
    fractional_bits: int,
    input_encoding: str,
    compute_opcode: str,
) -> ProjectionExport:
    """Quantize one Conv-BN pair and generate its exact post-accumulator fields."""

    integer_weight, weight_scale = quantize_weight_tensor(convolution.weight)
    output_channels = convolution.out_channels
    matrix = integer_weight.reshape(output_channels, -1).transpose(0, 1).contiguous()
    batch_norm_scale = batch_norm.weight.detach() / torch.sqrt(
        batch_norm.running_var.detach() + batch_norm.eps)
    if convolution.bias is None:
        convolution_bias = torch.zeros_like(batch_norm.running_mean)
    else:
        convolution_bias = convolution.bias.detach()
    folded_bias = (
        (convolution_bias - batch_norm.running_mean.detach()) * batch_norm_scale
        + batch_norm.bias.detach()
    )
    output_scale = 1 << fractional_bits
    accumulator_factor = (
        input_scale * weight_scale * batch_norm_scale * output_scale)
    multiplier: list[int] = []
    binary_shift: list[int] = []
    represented: list[float] = []
    for value in accumulator_factor.tolist():
        q15, shift, represented_value = quantize_flexible_scale(float(value))
        multiplier.append(q15)
        binary_shift.append(shift)
        represented.append(represented_value)
    bias_integer = torch.round(folded_bias * output_scale).to(torch.int32)
    factor_error = torch.tensor(represented) - accumulator_factor.cpu()

    sections = (
        writer.append(f"{name}.weight_k_by_n_int8", matrix.cpu().numpy()),
        writer.append(f"{name}.bias_q{fractional_bits}", bias_integer.cpu().numpy()),
        writer.append(f"{name}.multiplier_q15", np.asarray(multiplier, dtype="<i2")),
        writer.append(f"{name}.binary_point_shift", np.asarray(binary_shift, dtype=np.int8)),
    )
    return ProjectionExport(
        name=name,
        source_weight_shape=tuple(convolution.weight.shape),
        hardware_matrix_shape=tuple(matrix.shape),
        input_encoding=input_encoding,
        compute_opcode=compute_opcode,
        output_fractional_bits=fractional_bits,
        weight_scale=weight_scale,
        accumulator_factor_minimum=float(accumulator_factor.amin().item()),
        accumulator_factor_maximum=float(accumulator_factor.amax().item()),
        represented_factor_maximum_error=float(factor_error.abs().amax().item()),
        folded_bias_minimum=int(bias_integer.amin().item()),
        folded_bias_maximum=int(bias_integer.amax().item()),
        sections=sections,
    )


def projection_pairs(model: Spikingformer) -> list[tuple[str, nn.Module, nn.Module]]:
    pairs: list[tuple[str, nn.Module, nn.Module]] = [
        (f"patch_embed.block{index}",
         getattr(model.patch_embed, f"block{index}_conv"),
         getattr(model.patch_embed, f"block{index}_bn"))
        for index in range(5)
    ]
    for index, block in enumerate(model.block):
        for name in ("q", "k", "v", "proj"):
            pairs.append((
                f"block.{index}.attn.{name}",
                getattr(block.attn, f"{name}_conv"),
                getattr(block.attn, f"{name}_bn"),
            ))
        for layer in (1, 2):
            pairs.append((
                f"block.{index}.mlp.fc{layer}",
                getattr(block.mlp, f"mlp{layer}_conv"),
                getattr(block.mlp, f"mlp{layer}_bn"),
            ))
    return pairs


def export_model(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    fractional_bits: int = 10,
    head_activation_scale: float = 0.125,
) -> dict[str, object]:
    loaded = load_checkpoint(checkpoint_path, device="cpu")
    model = loaded.model
    writer = AlignedBinaryWriter()
    input_probe = torch.zeros(1, 3, 1, 1)
    _, image_scale, image_zero_point = quantize_cifar_input_u8(input_probe)
    projections: list[ProjectionExport] = []
    for name, convolution, batch_norm in projection_pairs(model):
        first_layer = name == "patch_embed.block0"
        projections.append(export_projection(
            writer,
            name=name,
            convolution=convolution,
            batch_norm=batch_norm,
            input_scale=image_scale if first_layer else 1.0,
            fractional_bits=fractional_bits,
            input_encoding=(
                f"asymmetric uint8 zero_point={image_zero_point} scale={image_scale:.12g}"
                if first_layer else "binary spike 0/1"),
            compute_opcode=("COMPUTE_DENSE_OUTER" if first_layer else "COMPUTE_BINARY"),
        ))

    head_weight, head_weight_scale = quantize_weight_tensor(model.head.weight)
    head_matrix = head_weight.transpose(0, 1).contiguous().cpu().numpy()
    head_accumulator_scale = head_activation_scale * head_weight_scale
    head_bias = torch.round(
        model.head.bias.detach() / head_accumulator_scale).to(torch.int32).cpu().numpy()
    head_sections = (
        writer.append("head.weight_k_by_n_int8", head_matrix),
        writer.append("head.bias_accumulator_units", head_bias),
    )
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / "spikingformer_neuraedge_weights.bin"
    binary_path.write_bytes(writer.data)
    binary_hash = hashlib.sha256(writer.data).hexdigest()
    manifest: dict[str, object] = {
        "format": "neuraedge-spikingformer-parameters-v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": loaded.epoch,
        "checkpoint_metric_percent": loaded.metric,
        "binary_file": binary_path.name,
        "binary_bytes": len(writer.data),
        "binary_sha256": binary_hash,
        "alignment_bytes": ALIGNMENT,
        "fixed_point": {
            "activation_fractional_bits": fractional_bits,
            "activation_scale": 1 << fractional_bits,
            "lif_membrane_bits": 24,
            "lif_current_bits": 16,
            "image_input_scale": image_scale,
            "image_input_zero_point": image_zero_point,
            "head_activation_scale": head_activation_scale,
        },
        "projections": [asdict(item) for item in projections],
        "head": {
            "source_weight_shape": list(model.head.weight.shape),
            "hardware_matrix_shape": list(head_matrix.shape),
            "activation_encoding": "signed int8 encoded as uint8 with zero point 128",
            "weight_encoding": "signed int8 encoded as uint8 with zero point 128",
            "compute_opcode": "COMPUTE_DENSE_DOT8",
            "weight_scale": head_weight_scale,
            "accumulator_scale": head_accumulator_scale,
            "bias_minimum": int(head_bias.min()),
            "bias_maximum": int(head_bias.max()),
            "sections": [asdict(item) for item in head_sections],
        },
        "sections": [asdict(item) for item in writer.sections],
    }
    manifest_path = output_dir / "spikingformer_neuraedge_weights.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fractional-bits", type=int, default=10)
    parser.add_argument("--head-activation-scale", type=float, default=0.125)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    exported = export_model(
        arguments.checkpoint.resolve(),
        arguments.output_dir.resolve(),
        fractional_bits=arguments.fractional_bits,
        head_activation_scale=arguments.head_activation_scale,
    )
    print(json.dumps({
        "binary_file": exported["binary_file"],
        "binary_bytes": exported["binary_bytes"],
        "binary_sha256": exported["binary_sha256"],
        "projection_count": len(exported["projections"]),
    }, indent=2))
