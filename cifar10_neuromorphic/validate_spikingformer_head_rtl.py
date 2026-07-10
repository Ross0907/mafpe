from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


NUM_CLASSES = 10
FEATURE_DIM = 384
INFER_SKIP_ABS_THRESH = 2048


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def signed_hex_to_int(token: str, bits: int) -> int:
    value = int(token.strip(), 16)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def load_mem(path: Path, bits: int) -> list[int]:
    return [signed_hex_to_int(line, bits) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def fixed_reference_from_mem(rtl_dir: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    features_flat = np.array(load_mem(rtl_dir / "cifar10_spikingformer_head_features.mem", 16), dtype=np.int64)
    weights = np.array(load_mem(rtl_dir / "cifar10_spikingformer_head_weights.mem", 16), dtype=np.int64).reshape(NUM_CLASSES, FEATURE_DIM)
    bias = np.array(load_mem(rtl_dir / "cifar10_spikingformer_head_bias.mem", 64), dtype=np.int64)
    labels = np.array([int(line.strip(), 16) for line in (rtl_dir / "cifar10_spikingformer_head_labels.mem").read_text(encoding="ascii").splitlines() if line.strip()], dtype=np.int64)
    features = features_flat[: sample_count * FEATURE_DIM].reshape(sample_count, FEATURE_DIM)
    if INFER_SKIP_ABS_THRESH > 0:
        skipped = np.abs(features) <= INFER_SKIP_ABS_THRESH
        features = features.copy()
        features[skipped] = 0
        skip_fraction = float(skipped.mean())
    else:
        skip_fraction = 0.0
    scores = features @ weights.T + bias
    preds = scores.argmax(axis=1)
    return labels[:sample_count], scores, preds, skip_fraction


def write_report(args: argparse.Namespace) -> dict[str, object]:
    rtl_rows = load_rows(args.results_csv)

    labels = np.array([int(r["label"]) for r in rtl_rows], dtype=np.int64)
    preds = np.array([int(r["predicted"]) for r in rtl_rows], dtype=np.int64)
    correct = preds == labels
    rtl_logits = np.array([[int(r[f"score{c}"]) for c in range(NUM_CLASSES)] for r in rtl_rows], dtype=np.int64)
    labels_ref, ref_logits, pred_ref, skip_fraction = fixed_reference_from_mem(args.rtl_dir, len(rtl_rows))
    if not np.array_equal(labels, labels_ref):
        raise ValueError("RTL result labels do not match labels memory")
    error = rtl_logits - ref_logits
    mismatch_count = int((error != 0).sum())
    pred_mismatch = int((preds != pred_ref).sum())

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for label, pred in zip(labels, preds):
        confusion[label, pred] += 1

    metrics = {
        "samples": int(len(rtl_rows)),
        "accuracy": float(correct.mean()) if len(rtl_rows) else 0.0,
        "correct": int(correct.sum()),
        "score_mismatch_count": mismatch_count,
        "score_exact_match": mismatch_count == 0,
        "prediction_mismatch_count": pred_mismatch,
        "prediction_exact_match": pred_mismatch == 0,
        "max_abs_score_error": int(np.abs(error).max()) if error.size else 0,
        "infer_skip_abs_thresh": INFER_SKIP_ABS_THRESH,
        "feature_mac_skip_fraction": skip_fraction,
        "results_csv": str(args.results_csv),
        "reference": "recomputed from current RTL .mem files",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spikingformer_head_rtl_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 5), dpi=160)
    plt.imshow(error, aspect="auto", cmap="coolwarm")
    plt.xlabel("class")
    plt.ylabel("sample")
    plt.colorbar(label="RTL score - fixed reference")
    plt.tight_layout()
    plt.savefig(args.out_dir / "spikingformer_head_rtl_score_error.png")
    plt.close()

    norm_confusion = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(7, 6), dpi=160)
    plt.imshow(norm_confusion, cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("predicted")
    plt.ylabel("label")
    plt.colorbar(label="row-normalized count")
    plt.tight_layout()
    plt.savefig(args.out_dir / "spikingformer_head_rtl_confusion.png")
    plt.close()

    with PdfPages(args.out_dir / "spikingformer_head_rtl_validation.pdf") as pdf:
        fig = plt.figure(figsize=(10, 7), dpi=160)
        fig.text(0.06, 0.92, "Spikingformer Head RTL Validation", fontsize=18, weight="bold")
        lines = [
            f"Samples: {metrics['samples']}",
            f"RTL top-1: {metrics['accuracy'] * 100:.2f}%",
            f"Correct: {metrics['correct']} / {metrics['samples']}",
            f"Score exact match: {metrics['score_exact_match']}",
            f"Score mismatches: {metrics['score_mismatch_count']}",
            f"Prediction exact match: {metrics['prediction_exact_match']}",
            f"Prediction mismatches: {metrics['prediction_mismatch_count']}",
            f"Max absolute score error: {metrics['max_abs_score_error']}",
        ]
        fig.text(0.06, 0.82, "\n".join(lines), fontsize=12, va="top")
        pdf.savefig(fig)
        plt.close(fig)
        for name in ["spikingformer_head_rtl_score_error.png", "spikingformer_head_rtl_confusion.png"]:
            img = plt.imread(args.out_dir / name)
            fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    return metrics


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Validate Spikingformer head RTL against exported fixed-point reference.")
    parser.add_argument("--results-csv", type=Path, default=root / "cifar10_neuromorphic" / "cifar10_spikingformer_head_results.csv")
    parser.add_argument("--rtl-dir", type=Path, default=root / "cifar10_neuromorphic" / "rtl")
    parser.add_argument("--out-dir", type=Path, default=root / "cifar10_neuromorphic" / "outputs_spikingformer_head_accelerator")
    return parser.parse_args()


def main() -> None:
    metrics = write_report(parse_args())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
