from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
from timm.data import create_transform
from timm.models import create_model
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
FEATURE_DIM = 384


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_runtime_dlls() -> None:
    env_root = repo_root() / "external_research" / "spikingformer_py310"
    for path in (
        env_root / "Lib" / "site-packages" / "torch" / "lib",
        env_root / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin",
        env_root / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
    ):
        if path.exists():
            try:
                os.add_dll_directory(str(path))
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def import_upstream_model() -> None:
    upstream = repo_root() / "external_research" / "Spikingformer" / "cifar10"
    if not upstream.exists():
        raise FileNotFoundError(f"Missing upstream Spikingformer source: {upstream}")
    sys.path.insert(0, str(upstream))
    import model  # noqa: F401


def signed_hex(value: int, bits: int) -> str:
    return f"{value & ((1 << bits) - 1):0{bits // 4}x}"


def choose_scale(values: torch.Tensor, bits: int) -> float:
    qmax = float((1 << (bits - 1)) - 1)
    max_abs = float(values.detach().abs().max().cpu())
    return max(max_abs / qmax, 1.0e-12)


def quantize_signed(values: torch.Tensor, scale: float, bits: int) -> torch.Tensor:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    return torch.round(values / scale).clamp(qmin, qmax).to(torch.int64)


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    clean: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean[key.removeprefix("module.")] = value
    return clean


def build_model(device: torch.device) -> nn.Module:
    model = create_model(
        "Spikingformer",
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=FEATURE_DIM,
        num_heads=12,
        mlp_ratios=4,
        in_channels=3,
        num_classes=NUM_CLASSES,
        qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
        depths=4,
        sr_ratios=1,
        T=4,
    )
    return model.to(device)


def make_loader(args: argparse.Namespace) -> DataLoader:
    transform = create_transform(
        input_size=(3, 32, 32),
        is_training=False,
        interpolation="bicubic",
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        crop_pct=1.0,
    )
    dataset = datasets.CIFAR10(root=str(args.data_dir), train=False, download=True, transform=transform)
    if args.samples > 0:
        dataset = Subset(dataset, range(min(args.samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return nullcontext()


def collect_features(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    add_runtime_dlls()
    import_upstream_model()
    from spikingjelly.clock_driven import functional

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(device)
    model.load_state_dict(load_checkpoint(args.checkpoint), strict=True)
    model.eval()
    loader = make_loader(args)

    all_features: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device, args.amp):
                time_images = images.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)
                time_features = model.forward_features(time_images)
                features = time_features.mean(0)
                logits = model.head(features)
            functional.reset_net(model)
            all_features.append(features.detach().float().cpu())
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_features), torch.cat(all_logits), torch.cat(all_labels)


def fold_batch_norm(weight: torch.Tensor, bias: torch.Tensor | None, bn_prefix: str, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    out_channels = weight.shape[0]
    if bias is None:
        bias = torch.zeros(out_channels, dtype=weight.dtype)
    gamma = state[f"{bn_prefix}.weight"]
    beta = state[f"{bn_prefix}.bias"]
    mean = state[f"{bn_prefix}.running_mean"]
    var = state[f"{bn_prefix}.running_var"]
    scale = gamma / torch.sqrt(var + 1.0e-5)
    folded_weight = weight * scale.reshape(-1, *([1] * (weight.ndim - 1)))
    folded_bias = (bias - mean) * scale + beta
    return folded_weight, folded_bias


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | list[int]]:
    t = tensor.detach().float()
    return {
        "shape": list(t.shape),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
    }


def full_checkpoint_mapping(args: argparse.Namespace) -> dict[str, object]:
    state = load_checkpoint(args.checkpoint)
    layers: list[dict[str, object]] = []

    conv_bn_pairs = [
        ("patch_embed.block0_conv", "patch_embed.block0_bn"),
        ("patch_embed.block1_conv", "patch_embed.block1_bn"),
        ("patch_embed.block2_conv", "patch_embed.block2_bn"),
        ("patch_embed.block3_conv", "patch_embed.block3_bn"),
        ("patch_embed.block4_conv", "patch_embed.block4_bn"),
    ]
    for block in range(4):
        base = f"block.{block}"
        conv_bn_pairs.extend(
            [
                (f"{base}.attn.q_conv", f"{base}.attn.q_bn"),
                (f"{base}.attn.k_conv", f"{base}.attn.k_bn"),
                (f"{base}.attn.v_conv", f"{base}.attn.v_bn"),
                (f"{base}.attn.proj_conv", f"{base}.attn.proj_bn"),
                (f"{base}.mlp.mlp1_conv", f"{base}.mlp.mlp1_bn"),
                (f"{base}.mlp.mlp2_conv", f"{base}.mlp.mlp2_bn"),
            ]
        )

    total_weights = 0
    total_biases = 0
    for conv_prefix, bn_prefix in conv_bn_pairs:
        weight = state[f"{conv_prefix}.weight"].float()
        bias = state.get(f"{conv_prefix}.bias")
        if bias is not None:
            bias = bias.float()
        folded_weight, folded_bias = fold_batch_norm(weight, bias, bn_prefix, state)
        total_weights += folded_weight.numel()
        total_biases += folded_bias.numel()
        layers.append(
            {
                "name": conv_prefix,
                "kind": "conv_or_pointwise_with_batchnorm_folded",
                "weight": tensor_stats(folded_weight),
                "bias": tensor_stats(folded_bias),
                "weight_values": int(folded_weight.numel()),
                "bias_values": int(folded_bias.numel()),
            }
        )

    head_weight = state["head.weight"].float()
    head_bias = state["head.bias"].float()
    total_weights += head_weight.numel()
    total_biases += head_bias.numel()
    layers.append(
        {
            "name": "head",
            "kind": "linear_classifier",
            "weight": tensor_stats(head_weight),
            "bias": tensor_stats(head_bias),
            "weight_values": int(head_weight.numel()),
            "bias_values": int(head_bias.numel()),
        }
    )

    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_metric_top1": None,
        "checkpoint_metric_note": "Current measured accuracy is written after export as exported_float_top1_pct and exported_fixed_top1_pct.",
        "source_model": "external_research/Spikingformer/cifar10/model.py",
        "topology": {
            "time_steps": 4,
            "image": [3, 32, 32],
            "tokenizer": "5 convolution stages with LIF gates after stages 0-3 and 2 max-pool stages",
            "transformer_blocks": 4,
            "embed_dim": FEATURE_DIM,
            "attention_heads": 12,
            "mlp_ratio": 4,
            "classifier": [FEATURE_DIM, NUM_CLASSES],
        },
        "mapped_layers": layers,
        "parameter_counts": {
            "weights": int(total_weights),
            "biases": int(total_biases),
            "total": int(total_weights + total_biases),
            "int8_weight_bytes": int(total_weights),
            "int16_weight_bytes": int(total_weights * 2),
            "float32_checkpoint_bytes": int((total_weights + total_biases) * 4),
        },
        "implementation_note": (
            "The shipped head RTL consumes fixed-point Spikingformer features and the trained classifier weights. "
            "The manifest above enumerates every trained backbone tensor that must be implemented by a DDR-backed "
            "streaming convolution/attention engine for a full bitstream-resident Spikingformer."
        ),
    }


def write_head_memories(
    args: argparse.Namespace,
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, object]:
    state = load_checkpoint(args.checkpoint)
    head_weight = state["head.weight"].float()
    head_bias = state["head.bias"].float()

    feature_scale = choose_scale(features, args.feature_bits)
    weight_scale = choose_scale(head_weight, args.weight_bits)
    product_scale = feature_scale * weight_scale
    features_q = quantize_signed(features, feature_scale, args.feature_bits)
    weights_q = quantize_signed(head_weight, weight_scale, args.weight_bits)
    bias_q = torch.round(head_bias / product_scale).to(torch.int64)
    features_eff_q = features_q.clone()
    if args.infer_skip_abs_thresh > 0:
        features_eff_q[features_eff_q.abs() <= args.infer_skip_abs_thresh] = 0
    logits_q = features_eff_q @ weights_q.t() + bias_q
    pred_float = logits.argmax(dim=1)
    pred_q = logits_q.argmax(dim=1)
    correct_float = pred_float.eq(labels)
    correct_q = pred_q.eq(labels)

    rtl_dir = args.rtl_dir
    rtl_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (rtl_dir / "cifar10_spikingformer_head_features.mem").open("w", encoding="ascii") as fh:
        for row in features_q.tolist():
            for value in row:
                fh.write(signed_hex(int(value), args.feature_bits) + "\n")
    with (rtl_dir / "cifar10_spikingformer_head_weights.mem").open("w", encoding="ascii") as fh:
        for row in weights_q.tolist():
            for value in row:
                fh.write(signed_hex(int(value), args.weight_bits) + "\n")
    with (rtl_dir / "cifar10_spikingformer_head_bias.mem").open("w", encoding="ascii") as fh:
        for value in bias_q.tolist():
            fh.write(signed_hex(int(value), 64) + "\n")
    with (rtl_dir / "cifar10_spikingformer_head_labels.mem").open("w", encoding="ascii") as fh:
        for value in labels.tolist():
            fh.write(f"{int(value):01x}\n")
    params = [
        f"`define CIFAR10_SF_HEAD_NUM_SAMPLES {features.shape[0]}",
        f"`define CIFAR10_SF_HEAD_FEATURE_DIM {FEATURE_DIM}",
        f"`define CIFAR10_SF_HEAD_NUM_CLASSES {NUM_CLASSES}",
        f"`define CIFAR10_SF_HEAD_FEATURE_BITS {args.feature_bits}",
        f"`define CIFAR10_SF_HEAD_WEIGHT_BITS {args.weight_bits}",
        "`define CIFAR10_SF_HEAD_ACC_BITS 64",
        f"`define CIFAR10_SF_HEAD_TRACE_SAMPLES {args.trace_samples}",
        f"`define CIFAR10_SF_HEAD_INFER_SKIP_ABS_THRESH {args.infer_skip_abs_thresh}",
    ]
    (rtl_dir / "cifar10_spikingformer_head_params.svh").write_text("\n".join(params) + "\n", encoding="ascii")

    fixed_ref = out_dir / "spikingformer_head_fixed_reference.csv"
    with fixed_ref.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["sample", "label", "float_pred", "fixed_pred", "float_correct", "fixed_correct"]
        fieldnames.extend([f"logit_q{c}" for c in range(NUM_CLASSES)])
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(features.shape[0]):
            row = {
                "sample": idx,
                "label": int(labels[idx]),
                "float_pred": int(pred_float[idx]),
                "fixed_pred": int(pred_q[idx]),
                "float_correct": int(correct_float[idx]),
                "fixed_correct": int(correct_q[idx]),
            }
            for cls in range(NUM_CLASSES):
                row[f"logit_q{cls}"] = int(logits_q[idx, cls])
            writer.writerow(row)

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for label, pred in zip(labels.numpy(), pred_q.numpy()):
        confusion[int(label), int(pred)] += 1

    metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "samples": int(features.shape[0]),
        "feature_dim": FEATURE_DIM,
        "float_accuracy": float(correct_float.float().mean()),
        "fixed_accuracy": float(correct_q.float().mean()),
        "float_fixed_prediction_agreement": float(pred_float.eq(pred_q).float().mean()),
        "feature_scale": feature_scale,
        "weight_scale": weight_scale,
        "product_scale": product_scale,
        "feature_bits": args.feature_bits,
        "weight_bits": args.weight_bits,
        "bias_bits": 64,
        "infer_skip_abs_thresh": args.infer_skip_abs_thresh,
        "feature_mac_skip_fraction": float((features_q.abs() <= args.infer_skip_abs_thresh).float().mean()) if args.infer_skip_abs_thresh > 0 else 0.0,
        "confusion": confusion.tolist(),
        "files": {
            "features": str(rtl_dir / "cifar10_spikingformer_head_features.mem"),
            "weights": str(rtl_dir / "cifar10_spikingformer_head_weights.mem"),
            "bias": str(rtl_dir / "cifar10_spikingformer_head_bias.mem"),
            "labels": str(rtl_dir / "cifar10_spikingformer_head_labels.mem"),
            "params": str(rtl_dir / "cifar10_spikingformer_head_params.svh"),
            "fixed_reference": str(fixed_ref),
        },
    }
    (out_dir / "spikingformer_head_mapping_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_report(args: argparse.Namespace, metrics: dict[str, object], mapping: dict[str, object]) -> None:
    out_dir = args.out_dir
    confusion = np.array(metrics["confusion"], dtype=np.float64)
    norm_confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1.0)

    plt.figure(figsize=(7, 6), dpi=160)
    plt.imshow(norm_confusion, cmap="viridis", vmin=0, vmax=1)
    plt.xticks(range(NUM_CLASSES), CIFAR10_CLASSES, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), CIFAR10_CLASSES)
    plt.colorbar(label="row-normalized count")
    plt.tight_layout()
    plt.savefig(out_dir / "spikingformer_head_fixed_confusion.png")
    plt.close()

    layer_names = [layer["name"] for layer in mapping["mapped_layers"]]
    layer_counts = [layer["weight_values"] for layer in mapping["mapped_layers"]]
    plt.figure(figsize=(12, 5), dpi=160)
    plt.bar(range(len(layer_counts)), layer_counts)
    plt.yscale("log")
    plt.xticks(range(len(layer_names)), layer_names, rotation=90, fontsize=6)
    plt.ylabel("trained weight values (log scale)")
    plt.tight_layout()
    plt.savefig(out_dir / "spikingformer_weight_inventory.png")
    plt.close()

    with PdfPages(out_dir / "spikingformer_head_accelerator_report.pdf") as pdf:
        fig = plt.figure(figsize=(10, 7), dpi=160)
        fig.text(0.06, 0.93, "Trained Spikingformer Weight Mapping", fontsize=18, weight="bold")
        lines = [
            f"Checkpoint: {Path(str(metrics['checkpoint'])).name}",
            f"Samples exported: {metrics['samples']}",
            f"Float top-1 on exported set: {float(metrics['float_accuracy']) * 100:.2f}%",
            f"Fixed-point head top-1: {float(metrics['fixed_accuracy']) * 100:.2f}%",
            f"Float/fixed prediction agreement: {float(metrics['float_fixed_prediction_agreement']) * 100:.2f}%",
            f"Feature gate: abs(feature) <= {metrics['infer_skip_abs_thresh']} maps to zero",
            f"Feature-MAC skip opportunity: {float(metrics['feature_mac_skip_fraction']) * 100:.2f}%",
            f"Feature scale: {metrics['feature_scale']:.8g}",
            f"Weight scale: {metrics['weight_scale']:.8g}",
            f"Mapped trained parameters inventoried: {mapping['parameter_counts']['total']:,}",
            "RTL scope in this run: trained 384-to-10 Spikingformer classifier head over exported backbone features.",
            "Full backbone scope captured in manifest: tokenizer convs, q/k/v/proj attention, MLP, BN-folded weights.",
        ]
        fig.text(0.06, 0.83, "\n".join(lines), fontsize=11, va="top")
        pdf.savefig(fig)
        plt.close(fig)
        for name in ["spikingformer_head_fixed_confusion.png", "spikingformer_weight_inventory.png"]:
            img = plt.imread(out_dir / name)
            fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Export trained Spikingformer CIFAR-10 weights/features for RTL acceleration.")
    parser.add_argument("--checkpoint", type=Path, default=root / "cifar10_neuromorphic" / "outputs_upstream_spikingformer" / "spikingformer_4_384_400e" / "model_best.pth.tar")
    parser.add_argument("--data-dir", type=Path, default=root / "datasets")
    parser.add_argument("--rtl-dir", type=Path, default=root / "cifar10_neuromorphic" / "rtl")
    parser.add_argument("--out-dir", type=Path, default=root / "cifar10_neuromorphic" / "outputs_spikingformer_head_accelerator")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--trace-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-bits", type=int, default=16)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--infer-skip-abs-thresh", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, logits, labels = collect_features(args)
    mapping = full_checkpoint_mapping(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_head_memories(args, features, logits, labels)
    mapping["exported_float_top1_pct"] = float(metrics["float_accuracy"]) * 100.0
    mapping["exported_fixed_top1_pct"] = float(metrics["fixed_accuracy"]) * 100.0
    mapping["infer_skip_abs_thresh"] = int(metrics["infer_skip_abs_thresh"])
    mapping["feature_mac_skip_fraction_pct"] = float(metrics["feature_mac_skip_fraction"]) * 100.0
    (args.out_dir / "spikingformer_full_checkpoint_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    write_report(args, metrics, mapping)
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"confusion", "files"}}, indent=2))
    print(f"outputs={args.out_dir}")


if __name__ == "__main__":
    main()
