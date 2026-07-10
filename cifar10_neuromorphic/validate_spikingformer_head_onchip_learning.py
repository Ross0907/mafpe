from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


NUM_CLASSES = 10
FEATURE_DIM = 384
LEARN_SHIFT = 14
BIAS_LEARN_DELTA = 1
LEARN_MARGIN_Q = 268435456
INFER_SKIP_ABS_THRESH = 2048


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def signed_hex_to_int(token: str, bits: int) -> int:
    value = int(token.strip(), 16)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def load_mem(path: Path, bits: int) -> list[int]:
    return [signed_hex_to_int(line, bits) for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def clamp_i16(value: int) -> int:
    return max(-32768, min(32767, value))


def effective_features(features: np.ndarray) -> np.ndarray:
    if INFER_SKIP_ABS_THRESH <= 0:
        return features
    out = features.copy()
    out[np.abs(out) <= INFER_SKIP_ABS_THRESH] = 0
    return out


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def simulate_online_learning(rtl_dir: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features_flat = load_mem(rtl_dir / "cifar10_spikingformer_head_features.mem", 16)
    labels = np.array([int(line.strip(), 16) for line in (rtl_dir / "cifar10_spikingformer_head_labels.mem").read_text(encoding="ascii").splitlines() if line.strip()], dtype=np.int64)
    weights = np.array(load_mem(rtl_dir / "cifar10_spikingformer_head_weights.mem", 16), dtype=np.int64).reshape(NUM_CLASSES, FEATURE_DIM)
    bias = np.array(load_mem(rtl_dir / "cifar10_spikingformer_head_bias.mem", 64), dtype=np.int64)
    features = np.array(features_flat[: sample_count * FEATURE_DIM], dtype=np.int64).reshape(sample_count, FEATURE_DIM)

    scores_out = np.zeros((sample_count, NUM_CLASSES), dtype=np.int64)
    preds = np.zeros(sample_count, dtype=np.int64)
    update_applied = np.zeros(sample_count, dtype=np.int64)
    update_count = np.zeros(sample_count, dtype=np.int64)
    count = 0

    for sample_idx in range(sample_count):
        x = effective_features(features[sample_idx])
        scores = bias + weights @ x
        pred = int(scores.argmax())
        label = int(labels[sample_idx])
        scores_out[sample_idx] = scores
        preds[sample_idx] = pred
        if pred != label and (int(scores[pred]) - int(scores[label])) <= LEARN_MARGIN_Q:
            delta = np.array([int(v) >> LEARN_SHIFT for v in x], dtype=np.int64)
            weights[label] = np.array([clamp_i16(int(w) + int(d)) for w, d in zip(weights[label], delta)], dtype=np.int64)
            weights[pred] = np.array([clamp_i16(int(w) - int(d)) for w, d in zip(weights[pred], delta)], dtype=np.int64)
            bias[label] += BIAS_LEARN_DELTA
            bias[pred] -= BIAS_LEARN_DELTA
            count += 1
            update_applied[sample_idx] = 1
        update_count[sample_idx] = count
    return scores_out, preds, update_applied, update_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate on-chip learning RTL against fixed-point online-learning golden model.")
    parser.add_argument("--rtl-dir", type=Path, default=repo_root() / "cifar10_neuromorphic" / "rtl")
    parser.add_argument("--results-csv", type=Path, default=repo_root() / "cifar10_neuromorphic" / "cifar10_spikingformer_head_onchip_learning_results.csv")
    parser.add_argument("--out-dir", type=Path, default=repo_root() / "cifar10_neuromorphic" / "outputs_spikingformer_head_accelerator")
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    scores_ref, preds_ref, updates_ref, update_count_ref = simulate_online_learning(args.rtl_dir, len(rows))
    scores_rtl = np.array([[int(row[f"score{c}"]) for c in range(NUM_CLASSES)] for row in rows], dtype=np.int64)
    preds_rtl = np.array([int(row["predicted"]) for row in rows], dtype=np.int64)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    updates_rtl = np.array([int(row["update_applied"]) for row in rows], dtype=np.int64)
    update_count_rtl = np.array([int(row["update_count"]) for row in rows], dtype=np.int64)

    score_error = scores_rtl - scores_ref
    metrics = {
        "samples": int(len(rows)),
        "accuracy": float((preds_rtl == labels).mean()) if rows else 0.0,
        "correct": int((preds_rtl == labels).sum()),
        "updates": int(updates_rtl.sum()),
        "score_mismatch_count": int((score_error != 0).sum()),
        "max_abs_score_error": int(np.abs(score_error).max()) if score_error.size else 0,
        "prediction_mismatch_count": int((preds_rtl != preds_ref).sum()),
        "update_flag_mismatch_count": int((updates_rtl != updates_ref).sum()),
        "update_count_mismatch_count": int((update_count_rtl != update_count_ref).sum()),
        "learn_shift": LEARN_SHIFT,
        "bias_learn_delta": BIAS_LEARN_DELTA,
        "learn_margin_q": LEARN_MARGIN_Q,
        "infer_skip_abs_thresh": INFER_SKIP_ABS_THRESH,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "spikingformer_head_onchip_learning_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
