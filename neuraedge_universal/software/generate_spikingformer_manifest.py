"""Generate the exact Spikingformer-to-NeuraEdge execution manifest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Literal

import torch


ROW_TILE = 8
COLUMN_TILE = 16
MAX_REDUCTION = 2048


@dataclass(frozen=True)
class MatrixSchedule:
    rows: int
    reduction: int
    columns: int
    repetitions: int
    arithmetic: Literal["u8_mac", "binary_spike_add", "dynamic_u8_mac"]
    mac_equivalents: int
    reduction_chunks: int
    compute_tiles: int
    commands: int
    input_stream_bytes: int
    output_stream_bytes: int


@dataclass(frozen=True)
class OperatorMapping:
    name: str
    source_operation: str
    input_shape_per_image: tuple[int, ...]
    output_shape_per_image: tuple[int, ...]
    hardware_primitive: str
    matrix: MatrixSchedule | None
    exact_optimization: str | None = None


def matrix_schedule(
    rows: int,
    reduction: int,
    columns: int,
    *,
    repetitions: int = 1,
    arithmetic: Literal["u8_mac", "binary_spike_add", "dynamic_u8_mac"],
) -> MatrixSchedule:
    """Count commands and stream bytes using the compiler's public tiling ABI."""

    if min(rows, reduction, columns, repetitions) <= 0:
        raise ValueError("all matrix dimensions and repetitions must be positive")
    total_commands = 0
    total_compute_tiles = 0
    total_input_bytes = 0
    total_output_bytes = 0
    chunks = 0
    for reduction_start in range(0, reduction, MAX_REDUCTION):
        chunk = min(MAX_REDUCTION, reduction - reduction_start)
        chunks += 1
        row_tiles = ceil(rows / ROW_TILE)
        column_tiles = ceil(columns / COLUMN_TILE)
        compute_tiles = row_tiles * column_tiles
        if rows == 1:
            issues = ceil(chunk / 8)
            activation_tile_bytes = issues * 8
            weight_tile_bytes = issues * 8 * 16
        else:
            activation_tile_bytes = chunk * ROW_TILE
            weight_tile_bytes = chunk * COLUMN_TILE
        input_broadcast_bytes = (
            row_tiles * activation_tile_bytes
            + compute_tiles * weight_tile_bytes
        )
        weight_broadcast_bytes = (
            column_tiles * weight_tile_bytes
            + compute_tiles * activation_tile_bytes
        )
        if input_broadcast_bytes <= weight_broadcast_bytes:
            load_commands = row_tiles + compute_tiles
            stream_bytes = input_broadcast_bytes
        else:
            load_commands = column_tiles + compute_tiles
            stream_bytes = weight_broadcast_bytes
        total_commands += load_commands + compute_tiles
        total_compute_tiles += compute_tiles
        total_input_bytes += stream_bytes
        # The current output ABI emits one 32-bit result in each 64-bit beat.
        total_output_bytes += rows * columns * 8
    return MatrixSchedule(
        rows=rows,
        reduction=reduction,
        columns=columns,
        repetitions=repetitions,
        arithmetic=arithmetic,
        mac_equivalents=rows * reduction * columns * repetitions,
        reduction_chunks=chunks * repetitions,
        compute_tiles=total_compute_tiles * repetitions,
        commands=total_commands * repetitions,
        input_stream_bytes=total_input_bytes * repetitions,
        output_stream_bytes=total_output_bytes * repetitions,
    )


def build_operator_manifest() -> list[OperatorMapping]:
    """Describe every learned and data-dependent operator for one image."""

    time_steps = 4
    channels = 384
    tokens = 64
    heads = 12
    head_dim = channels // heads
    operators: list[OperatorMapping] = []

    def matrix_operator(
        name: str,
        source: str,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        rows: int,
        reduction: int,
        columns: int,
        arithmetic: Literal["u8_mac", "binary_spike_add", "dynamic_u8_mac"],
        primitive: str,
        *,
        repetitions: int = 1,
        optimization: str | None = None,
    ) -> None:
        operators.append(OperatorMapping(
            name=name,
            source_operation=source,
            input_shape_per_image=input_shape,
            output_shape_per_image=output_shape,
            hardware_primitive=primitive,
            matrix=matrix_schedule(
                rows, reduction, columns,
                repetitions=repetitions,
                arithmetic=arithmetic,
            ),
            exact_optimization=optimization,
        ))

    matrix_operator(
        "patch_embed.block0", "Conv2d 3x3 + folded BatchNorm",
        (3, 32, 32), (time_steps, 48, 32, 32),
        32 * 32, 3 * 3 * 3, 48, "u8_mac", "dense tensor tile",
        optimization=(
            "Compute once and broadcast to four timesteps. The source repeats "
            "one unchanged image before this stateless operator, so this removes "
            "three of four executions with exact output equivalence."),
    )
    matrix_operator(
        "patch_embed.block1", "LIF + Conv2d 3x3 + folded BatchNorm",
        (time_steps, 48, 32, 32), (time_steps, 96, 32, 32),
        time_steps * 32 * 32, 48 * 3 * 3, 96,
        "binary_spike_add", "active-index sparse tensor tile")
    matrix_operator(
        "patch_embed.block2", "LIF + Conv2d 3x3 + folded BatchNorm",
        (time_steps, 96, 32, 32), (time_steps, 192, 32, 32),
        time_steps * 32 * 32, 96 * 3 * 3, 192,
        "binary_spike_add", "active-index sparse tensor tile")
    matrix_operator(
        "patch_embed.block3", "LIF + MaxPool2d + Conv2d 3x3 + folded BatchNorm",
        (time_steps, 192, 32, 32), (time_steps, 384, 16, 16),
        time_steps * 16 * 16, 192 * 3 * 3, 384,
        "binary_spike_add", "vector maximum + active-index sparse tensor tile")
    matrix_operator(
        "patch_embed.block4", "LIF + MaxPool2d + Conv2d 3x3 + folded BatchNorm",
        (time_steps, 384, 16, 16), (time_steps, 384, 8, 8),
        time_steps * 8 * 8, 384 * 3 * 3, 384,
        "binary_spike_add", "vector maximum + active-index sparse tensor tile")

    for block in range(4):
        prefix = f"block.{block}"
        for projection in ("q", "k", "v"):
            matrix_operator(
                f"{prefix}.attn.{projection}",
                f"LIF + {projection.upper()} Conv1d 1x1 + folded BatchNorm",
                (time_steps, channels, 8, 8),
                (time_steps, channels, tokens),
                time_steps * tokens, channels, channels,
                "binary_spike_add", "active-index sparse tensor tile")
        matrix_operator(
            f"{prefix}.attn.qk", "binary query times transposed binary key",
            (time_steps, heads, tokens, head_dim),
            (time_steps, heads, tokens, tokens),
            tokens, head_dim, tokens,
            "dynamic_u8_mac", "dense tensor tile",
            repetitions=time_steps * heads)
        matrix_operator(
            f"{prefix}.attn.av", "attention counts times binary value",
            (time_steps, heads, tokens, tokens),
            (time_steps, heads, tokens, head_dim),
            tokens, tokens, head_dim,
            "dynamic_u8_mac", "dense tensor tile",
            repetitions=time_steps * heads)
        matrix_operator(
            f"{prefix}.attn.proj", "LIF + projection Conv1d + folded BatchNorm",
            (time_steps, channels, tokens),
            (time_steps, channels, 8, 8),
            time_steps * tokens, channels, channels,
            "binary_spike_add", "active-index sparse tensor tile")
        matrix_operator(
            f"{prefix}.mlp.fc1", "LIF + 1x1 Conv2d + folded BatchNorm",
            (time_steps, channels, 8, 8),
            (time_steps, channels * 4, 8, 8),
            time_steps * tokens, channels, channels * 4,
            "binary_spike_add", "active-index sparse tensor tile")
        matrix_operator(
            f"{prefix}.mlp.fc2", "LIF + 1x1 Conv2d + folded BatchNorm",
            (time_steps, channels * 4, 8, 8),
            (time_steps, channels, 8, 8),
            time_steps * tokens, channels * 4, channels,
            "binary_spike_add", "active-index sparse tensor tile")
        operators.append(OperatorMapping(
            name=f"{prefix}.residuals",
            source_operation="two elementwise residual additions",
            input_shape_per_image=(time_steps, channels, 8, 8),
            output_shape_per_image=(time_steps, channels, 8, 8),
            hardware_primitive="saturating vector add",
            matrix=None,
        ))

    operators.append(OperatorMapping(
        name="global_average_pool",
        source_operation="mean over 64 tokens then mean over four timesteps",
        input_shape_per_image=(time_steps, channels, 8, 8),
        output_shape_per_image=(channels,),
        hardware_primitive="vector add followed by exact power-of-two shifts",
        matrix=None,
    ))
    matrix_operator(
        "head", "Linear classifier",
        (channels,), (10,), 1, channels, 10,
        "u8_mac", "dense Dot8 tensor tile")
    return operators


def generate(checkpoint_path: Path, output_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["state_dict"]
    parameter_tensors = {
        name: {
            "shape": list(value.shape),
            "elements": value.numel(),
            "source_dtype": str(value.dtype).replace("torch.", ""),
        }
        for name, value in state.items()
        if name.endswith(("weight", "bias"))
    }
    operators = build_operator_manifest()
    matrix_operators = [item.matrix for item in operators if item.matrix is not None]
    total_macs = sum(item.mac_equivalents for item in matrix_operators)
    total_commands = sum(item.commands for item in matrix_operators)
    total_input_bytes = sum(item.input_stream_bytes for item in matrix_operators)
    total_output_bytes = sum(item.output_stream_bytes for item in matrix_operators)
    trainable_parameters = sum(
        value.numel() for name, value in state.items()
        if name.endswith(("weight", "bias"))
    )
    int8_weight_elements = sum(
        value.numel() for name, value in state.items()
        if name.endswith("weight")
        and not ("norm1" in name or "norm2" in name or "_bn." in name)
    )
    folded_bias_channels = sum(
        value.shape[0] for name, value in state.items()
        if name.endswith("running_mean")
    ) + state["head.bias"].numel()
    report: dict[str, object] = {
        "format": "neuraedge-spikingformer-execution-manifest-v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_metric_percent": float(checkpoint.get("metric", float("nan"))),
        "model": {
            "time_steps": 4,
            "embed_dim": 384,
            "heads": 12,
            "head_dim": 32,
            "transformer_blocks": 4,
            "tokens": 64,
            "classes": 10,
            "trainable_parameter_elements": trainable_parameters,
            "source_parameter_bytes_fp32": trainable_parameters * 4,
            "mapped_int8_weight_bytes": int8_weight_elements,
            "mapped_int32_folded_bias_bytes": folded_bias_channels * 4,
        },
        "accelerator": {
            "row_tile": ROW_TILE,
            "column_tile": COLUMN_TILE,
            "maximum_reduction_per_command": MAX_REDUCTION,
            "packed_u8_macs_per_cycle": 128,
            "target_clock_hz": 200_000_000,
            "peak_dense_gmac_per_second": 25.6,
        },
        "per_image_totals": {
            "mac_equivalents_after_exact_temporal_hoist": total_macs,
            "tensor_commands": total_commands,
            "input_stream_bytes_without_inter_layer_spill": total_input_bytes,
            "output_stream_bytes_without_inter_layer_spill": total_output_bytes,
        },
        "operators": [asdict(item) for item in operators],
        "parameter_tensors": parameter_tensors,
        "semantic_notes": [
            "BatchNorm is folded into the preceding learned projection in evaluation mode.",
            "The checkpoint's LayerNorm parameters are retained for strict loading but are not called by its upstream forward graph.",
            "Binary-spike layers use signed int8 weights and addition/subtraction; MAC-equivalents are reported only for comparison with dense execution.",
            "QK and attention-value products have two dynamic operands and therefore use the exact dynamic-u8 path in this manifest.",
            "DDR traffic is not inferred from the summed tile streams because the final on-chip buffer schedule determines which operands are retained or spilled.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = generate(arguments.checkpoint.resolve(), arguments.output.resolve())
    print(json.dumps(result["per_image_totals"], indent=2))
