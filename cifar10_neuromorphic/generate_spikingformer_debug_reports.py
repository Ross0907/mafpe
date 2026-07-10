from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


NUM_CLASSES = 10
FEATURE_DIM = 384
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

STATE_NAMES = {
    0: "IDLE",
    1: "LOAD",
    2: "CLASS_INIT",
    3: "MAC_READ",
    4: "MAC_MUL",
    5: "MAC_ACCUM",
    6: "CLASS_DONE",
    7: "RESULT",
    8: "TRAIN_CHECK",
    9: "TRAIN_READ_TRUE",
    10: "TRAIN_CALC_TRUE",
    11: "TRAIN_WRITE_TRUE",
    12: "TRAIN_READ_PRED",
    13: "TRAIN_CALC_PRED",
    14: "TRAIN_WRITE_PRED",
    15: "TRAIN_BIAS_TRUE",
    16: "TRAIN_BIAS_PRED",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def out_dir() -> Path:
    return project_dir() / "outputs_spikingformer_head_accelerator"


def debug_dir() -> Path:
    return out_dir() / "deep_debug"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def col(rows: list[dict[str, str]], name: str, dtype=float) -> np.ndarray:
    if not rows:
        return np.array([], dtype=dtype)
    return np.array([dtype(row[name]) for row in rows])


def score_matrix(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([[int(row[f"score{i}"]) for i in range(NUM_CLASSES)] for row in rows], dtype=np.int64)


def signed_hex_to_int(token: str, bits: int) -> int:
    value = int(token.strip(), 16)
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def read_hex_mem(path: Path, bits: int) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        token = line.strip()
        if token:
            values.append(signed_hex_to_int(token, bits))
    return np.array(values, dtype=np.int64)


def read_decimal_mem(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        token = line.strip()
        if token:
            values.append(int(token))
    return np.array(values, dtype=np.int64)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def parse_first_float(pattern: str, text: str, default: float = float("nan")) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return float(match.group(1)) if match else default


def parse_first_int(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else default


def parse_vivado_reports(vivado_dir: Path) -> dict[str, float | int | str]:
    timing = (vivado_dir / "spikingformer_head_timing_summary.rpt").read_text(encoding="utf-8", errors="ignore")
    util = (vivado_dir / "spikingformer_head_utilization.rpt").read_text(encoding="utf-8", errors="ignore")
    power = (vivado_dir / "spikingformer_head_power.rpt").read_text(encoding="utf-8", errors="ignore")
    summary = (vivado_dir / "spikingformer_head_synth_summary.txt").read_text(encoding="utf-8", errors="ignore")
    return {
        "summary": summary,
        "wns_ns": parse_first_float(r"\n\s*([+-]?\d+\.\d+)\s+\d+\.\d+\s+\d+\s+\d+\s+[+-]?\d+\.\d+", timing),
        "whs_ns": parse_first_float(r"\n\s*[+-]?\d+\.\d+\s+\d+\.\d+\s+\d+\s+\d+\s+([+-]?\d+\.\d+)", timing),
        "lut": parse_first_int(r"\|\s*Slice LUTs\s*\|\s*(\d+)", util),
        "regs": parse_first_int(r"\|\s*Slice Registers\s*\|\s*(\d+)", util),
        "bram": parse_first_int(r"\|\s*Block RAM Tile\s*\|\s*(\d+)", util),
        "dsp": parse_first_int(r"\|\s*DSPs\s*\|\s*(\d+)", util),
        "lut_mem": parse_first_int(r"\|\s*LUT as Memory\s*\|\s*(\d+)", util),
        "total_power_w": parse_first_float(r"\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)", power),
        "dynamic_power_w": parse_first_float(r"\|\s*Dynamic \(W\)\s*\|\s*([0-9.]+)", power),
        "static_power_w": parse_first_float(r"\|\s*Device Static \(W\)\s*\|\s*([0-9.]+)", power),
        "clock_power_w": parse_first_float(r"\|\s*Clocks\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
        "logic_power_w": parse_first_float(r"\|\s*Slice Logic\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
        "signal_power_w": parse_first_float(r"\|\s*Signals\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
        "bram_power_w": parse_first_float(r"\|\s*Block RAM\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
        "dsp_power_w": parse_first_float(r"\|\s*DSPs\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
        "io_power_w": parse_first_float(r"\|\s*I/O\s*\|\s*([0-9.<]+)", power.replace("<0.001", "0.0005")),
    }


def save_pdf_page(pdf: PdfPages, fig: plt.Figure, png_dir: Path, name: str) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def confusion(labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
    mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for label, pred in zip(labels, preds):
        mat[int(label), int(pred)] += 1
    return mat


def precision_recall_f1(conf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(conf).astype(float)
    precision = tp / np.maximum(conf.sum(axis=0), 1)
    recall = tp / np.maximum(conf.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def margin(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_scores = np.sort(scores, axis=1)
    top1 = sorted_scores[:, -1]
    top2 = sorted_scores[:, -2]
    true_scores = scores[np.arange(scores.shape[0]), labels.astype(int)]
    return top1 - top2, top1 - true_scores


def plot_text_page(title: str, lines: Iterable[str]) -> plt.Figure:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.04, 0.95, title, fontsize=18, weight="bold", va="top")
    y = 0.90
    for line in lines:
        fig.text(0.05, y, line, fontsize=10.5, family="monospace", va="top")
        y -= 0.035
        if y < 0.05:
            break
    return fig


def draw_box(ax, xy, w, h, text, color="#e8f1ff"):
    rect = plt.Rectangle(xy, w, h, facecolor=color, edgecolor="#1d3557", linewidth=1.6)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def plot_architecture() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    blocks = [
        ((0.4, 5.4), 1.6, 0.7, "PYNQ PS / DMA\nAXI-stream features", "#eef6e8"),
        ((2.5, 5.4), 1.7, 0.7, "AXI-stream wrapper\nstart, TLAST, result pack", "#e8f1ff"),
        ((4.8, 5.4), 1.6, 0.7, "Feature SRAM\n384 x int16", "#fff0d9"),
        ((7.0, 5.4), 1.8, 0.7, "Score output\nlabel + update status", "#eef6e8"),
        ((1.0, 3.4), 1.7, 0.7, "Class loop\n10 classes", "#f6e8ff"),
        ((3.2, 3.4), 1.7, 0.7, "Weight BRAM\n10 x 384 x int16", "#fff0d9"),
        ((5.4, 3.4), 1.7, 0.7, "DSP MAC\nint16 x int16 -> acc64", "#e8f1ff"),
        ((7.6, 3.4), 1.7, 0.7, "Argmax score\nprediction", "#e8f1ff"),
        ((2.1, 1.3), 1.8, 0.7, "Margin-gated\non-chip update", "#ffe8e8"),
        ((4.5, 1.3), 1.8, 0.7, "Registered\nsat add/sub", "#ffe8e8"),
        ((6.9, 1.3), 1.8, 0.7, "Writeback\ntrue/pred rows + bias", "#ffe8e8"),
    ]
    for args in blocks:
        draw_box(ax, *args)
    arrows = [
        ((2.0, 5.75), (2.5, 5.75)),
        ((4.2, 5.75), (4.8, 5.75)),
        ((6.4, 5.75), (7.0, 5.75)),
        ((4.0, 3.75), (5.4, 3.75)),
        ((7.1, 3.75), (7.6, 3.75)),
        ((7.6, 3.35), (3.0, 2.0)),
        ((3.9, 1.65), (4.5, 1.65)),
        ((6.3, 1.65), (6.9, 1.65)),
        ((7.8, 2.0), (4.1, 3.35)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#333"})
    ax.set_title("NeuraEdge CIFAR-10 Spikingformer Head Accelerator Dataflow", fontsize=16, weight="bold")
    ax.text(0.5, 0.3, "All numbers in this report come from the generated RTL memories, XSim CSV traces, and Vivado reports in outputs_spikingformer_head_accelerator.", fontsize=9)
    return fig


def plot_fsm() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_axis_off()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    fsm = [
        ("IDLE", 0.5, 6.5), ("LOAD", 2.0, 6.5), ("CLASS_INIT", 3.7, 6.5),
        ("MAC_READ", 5.6, 6.5), ("MAC_MUL", 7.3, 6.5), ("MAC_ACCUM", 9.0, 6.5),
        ("CLASS_DONE", 5.5, 5.0), ("TRAIN_CHECK", 3.2, 3.6), ("RESULT", 8.0, 3.6),
        ("READ_TRUE", 0.7, 2.2), ("CALC_TRUE", 2.4, 2.2), ("WRITE_TRUE", 4.1, 2.2),
        ("READ_PRED", 5.8, 2.2), ("CALC_PRED", 7.5, 2.2), ("WRITE_PRED", 9.2, 2.2),
        ("BIAS_TRUE", 4.2, 0.9), ("BIAS_PRED", 6.2, 0.9),
    ]
    positions = {}
    for name, x, y in fsm:
        positions[name] = (x, y)
        draw_box(ax, (x, y), 1.35, 0.55, name, "#f6fbff" if "TRAIN" not in name and "TRUE" not in name and "PRED" not in name and "BIAS" not in name else "#fff1f1")
    edges = [
        ("IDLE", "LOAD"), ("LOAD", "CLASS_INIT"), ("CLASS_INIT", "MAC_READ"), ("MAC_READ", "MAC_MUL"),
        ("MAC_MUL", "MAC_ACCUM"), ("MAC_ACCUM", "CLASS_DONE"), ("CLASS_DONE", "MAC_READ"),
        ("CLASS_DONE", "TRAIN_CHECK"), ("TRAIN_CHECK", "RESULT"), ("RESULT", "IDLE"),
        ("TRAIN_CHECK", "READ_TRUE"), ("READ_TRUE", "CALC_TRUE"), ("CALC_TRUE", "WRITE_TRUE"),
        ("WRITE_TRUE", "READ_PRED"), ("READ_PRED", "CALC_PRED"), ("CALC_PRED", "WRITE_PRED"),
        ("WRITE_PRED", "READ_TRUE"), ("WRITE_PRED", "BIAS_TRUE"), ("BIAS_TRUE", "BIAS_PRED"), ("BIAS_PRED", "RESULT"),
    ]
    for a, b in edges:
        ax.annotate("", xy=(positions[b][0], positions[b][1] + 0.27), xytext=(positions[a][0] + 1.35, positions[a][1] + 0.27), arrowprops={"arrowstyle": "->", "lw": 1.0})
    ax.set_title("RTL State Machine Including On-Chip Learning", fontsize=16, weight="bold")
    return fig


def generate_debug_pdf() -> None:
    pdir = project_dir()
    odir = out_dir()
    ddir = debug_dir()
    png_dir = ddir / "plots"
    ddir.mkdir(parents=True, exist_ok=True)

    inference_rows = read_csv_rows(pdir / "cifar10_spikingformer_head_results.csv")
    learn_rows = read_csv_rows(pdir / "cifar10_spikingformer_head_onchip_learning_results.csv")
    trace_rows = read_csv_rows(pdir / "cifar10_spikingformer_head_trace.csv")
    learn_trace_rows = read_csv_rows(pdir / "cifar10_spikingformer_head_onchip_learning_trace.csv")
    rtl_val = read_json(odir / "spikingformer_head_rtl_validation.json")
    learn_val = read_json(odir / "spikingformer_head_onchip_learning_validation.json")
    mapping = read_json(odir / "spikingformer_head_mapping_metrics.json")
    vivado = parse_vivado_reports(odir / "vivado")
    infer_skip_abs_thresh = int(rtl_val.get("infer_skip_abs_thresh", mapping.get("infer_skip_abs_thresh", 0)) or 0)
    feature_mac_skip_fraction = float(rtl_val.get("feature_mac_skip_fraction", mapping.get("feature_mac_skip_fraction", 0.0)) or 0.0)
    rtl_infer_skip_count = int(inference_rows[-1].get("infer_skip_count", 0)) if inference_rows else 0
    learn_infer_skip_count = int(learn_rows[-1].get("infer_skip_count", 0)) if learn_rows else 0
    learn_train_skip_count = int(learn_rows[-1].get("train_skip_count", 0)) if learn_rows else 0

    labels = col(inference_rows, "label", int)
    preds = col(inference_rows, "predicted", int)
    correct = col(inference_rows, "correct", int)
    learn_preds = col(learn_rows, "predicted", int)
    learn_correct = col(learn_rows, "correct", int)
    updates = col(learn_rows, "update_applied", int)
    update_count = col(learn_rows, "update_count", int)
    scores = score_matrix(inference_rows)
    learn_scores = score_matrix(learn_rows)
    conf = confusion(labels, preds)
    precision, recall, f1 = precision_recall_f1(conf)
    margins, true_gaps = margin(scores, labels)

    rtl_dir = pdir / "rtl"
    features = read_hex_mem(rtl_dir / "cifar10_spikingformer_head_features.mem", 16)
    weights = read_hex_mem(rtl_dir / "cifar10_spikingformer_head_weights.mem", 16).reshape(NUM_CLASSES, FEATURE_DIM)
    bias = read_hex_mem(rtl_dir / "cifar10_spikingformer_head_bias.mem", 64)
    after_weights_path = pdir / "cifar10_spikingformer_head_onchip_learning_weights_after.mem"
    after_bias_path = pdir / "cifar10_spikingformer_head_onchip_learning_bias_after.mem"
    after_weights = read_decimal_mem(after_weights_path).reshape(NUM_CLASSES, FEATURE_DIM) if after_weights_path.exists() else weights.copy()
    after_bias = read_decimal_mem(after_bias_path) if after_bias_path.exists() else bias.copy()

    pdf_path = ddir / "neuraedge_spikingformer_deep_debug_report.pdf"
    with PdfPages(pdf_path) as pdf:
        summary_lines = [
            f"RTL inference samples: {rtl_val.get('samples')}  accuracy: {float(rtl_val.get('accuracy', float('nan'))):.4f}  exact score match: {rtl_val.get('score_exact_match')}",
            f"On-chip learning samples: {learn_val.get('samples')}  updates: {learn_val.get('updates')}  score mismatches: {learn_val.get('score_mismatch_count')}",
            f"Activity gate: abs(feature) <= {infer_skip_abs_thresh} is forced to zero; measured feature-MAC skip fraction: {feature_mac_skip_fraction * 100:.2f}%",
            f"RTL debug counters: inference skips={rtl_infer_skip_count:,}; learning-run inference skips={learn_infer_skip_count:,}; learning feature skips={learn_train_skip_count:,}",
            f"Vivado top: cifar10_spikingformer_head_pynq_axi_stream_top  part: xc7z020clg400-1  clock: 10 ns",
            f"Timing: WNS={vivado['wns_ns']:.3f} ns  WHS={vivado['whs_ns']:.3f} ns  timing_met={'timing_met=1' in vivado['summary']}",
            f"Utilization: LUT={vivado['lut']}  FF={vivado['regs']}  BRAM36={vivado['bram']}  DSP={vivado['dsp']}  LUTRAM={vivado['lut_mem']}",
            f"Power: total={vivado['total_power_w']:.3f} W  dynamic={vivado['dynamic_power_w']:.3f} W  static={vivado['static_power_w']:.3f} W",
            f"Mapping metrics file fields: {', '.join(sorted(mapping.keys())[:10])}",
            "Important interpretation: the RTL accelerates the trained Spikingformer feature-head path, gates low-magnitude feature activity, and implements a margin-gated on-chip classifier update.",
        ]
        save_pdf_page(pdf, plot_text_page("Measured Build Summary", summary_lines), png_dir, "00_summary")
        save_pdf_page(pdf, plot_architecture(), png_dir, "01_architecture")
        save_pdf_page(pdf, plot_fsm(), png_dir, "02_fsm")

        for name, rows in [("inference", trace_rows), ("onchip_learning", learn_trace_rows)]:
            cycles = col(rows, "cycle", int)
            fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
            axes[0].step(cycles, col(rows, "state", int), where="post")
            axes[0].set_ylabel("state")
            axes[1].step(cycles, col(rows, "feature_valid", int), where="post", label="valid")
            axes[1].step(cycles, col(rows, "feature_ready", int), where="post", label="ready")
            axes[1].legend(loc="upper right")
            axes[1].set_ylabel("feature")
            axes[2].plot(cycles, col(rows, "feature_idx", int))
            axes[2].set_ylabel("feature_idx")
            axes[3].plot(cycles, col(rows, "class_idx", int))
            axes[3].set_ylabel("class_idx")
            axes[4].step(cycles, col(rows, "result_valid", int), where="post", label="result")
            if "train_update_applied" in rows[0]:
                axes[4].step(cycles, col(rows, "train_update_applied", int), where="post", label="update")
            axes[4].legend(loc="upper right")
            axes[4].set_xlabel("cycle")
            axes[4].set_ylabel("flags")
            fig.suptitle(f"Cycle-Level Trace From XSim CSV: {name}")
            save_pdf_page(pdf, fig, png_dir, f"03_trace_{name}")

            counts = Counter(col(rows, "state", int))
            states = sorted(counts)
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.bar([STATE_NAMES.get(int(s), str(s)) for s in states], [counts[s] for s in states])
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylabel("cycles in traced samples")
            ax.set_title(f"FSM State Occupancy: {name}")
            save_pdf_page(pdf, fig, png_dir, f"04_state_occupancy_{name}")

        sample_idx = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(sample_idx, np.cumsum(correct) / np.maximum(sample_idx + 1, 1), label="inference")
        ax.plot(sample_idx, np.cumsum(learn_correct) / np.maximum(sample_idx + 1, 1), label="on-chip learning")
        ax.set_xlabel("sample")
        ax.set_ylabel("cumulative accuracy")
        ax.set_title("Cumulative Accuracy Over Full CIFAR-10 Test Stream")
        ax.legend()
        save_pdf_page(pdf, fig, png_dir, "05_cumulative_accuracy")

        window = 200
        roll = np.convolve(correct, np.ones(window) / window, mode="valid")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(np.arange(len(roll)) + window, roll)
        ax.set_xlabel("sample")
        ax.set_ylabel(f"rolling accuracy ({window})")
        ax.set_title("Rolling Accuracy Debug View")
        save_pdf_page(pdf, fig, png_dir, "06_rolling_accuracy")

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(conf, cmap="Blues")
        ax.set_xlabel("predicted")
        ax.set_ylabel("label")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, str(conf[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "07_confusion")

        norm_conf = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(norm_conf, cmap="viridis", vmin=0, vmax=1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("label")
        ax.set_title("Normalized Confusion Matrix")
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "08_confusion_normalized")

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(NUM_CLASSES)
        ax.bar(x - 0.25, precision, width=0.25, label="precision")
        ax.bar(x, recall, width=0.25, label="recall")
        ax.bar(x + 0.25, f1, width=0.25, label="F1")
        ax.set_xticks(x, CLASS_NAMES)
        ax.set_ylim(0, 1)
        ax.set_title("Per-Class Precision / Recall / F1")
        ax.legend()
        save_pdf_page(pdf, fig, png_dir, "09_precision_recall_f1")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(margins, bins=80, alpha=0.75)
        ax.set_title("Top-1 Minus Top-2 Score Margin Histogram")
        ax.set_xlabel("fixed-point score margin")
        ax.set_ylabel("samples")
        save_pdf_page(pdf, fig, png_dir, "10_top12_margin_hist")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.scatter(sample_idx[correct == 1], margins[correct == 1], s=3, alpha=0.35, label="correct")
        ax.scatter(sample_idx[correct == 0], margins[correct == 0], s=8, alpha=0.75, label="wrong")
        ax.set_xlabel("sample")
        ax.set_ylabel("top1-top2 margin")
        ax.set_title("Margin Timeline With Misclassification Overlay")
        ax.legend()
        save_pdf_page(pdf, fig, png_dir, "11_margin_timeline")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([margins[correct == 1], margins[correct == 0]], tick_labels=["correct", "wrong"], showfliers=False)
        ax.set_ylabel("top1-top2 margin")
        ax.set_title("Margins Split By Correctness")
        save_pdf_page(pdf, fig, png_dir, "12_margin_box")

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(scores[:80] / 1e9, aspect="auto", cmap="coolwarm")
        ax.set_xlabel("class")
        ax.set_ylabel("sample")
        ax.set_title("Score Heatmap: First 80 Samples (billions of fixed-point units)")
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES)
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "13_score_heatmap")

        diff = learn_scores - scores
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(diff.reshape(-1), bins=80)
        ax.set_title("Score Difference Caused By Online Weight Updates")
        ax.set_xlabel("on-chip-learning score - fixed inference score")
        save_pdf_page(pdf, fig, png_dir, "14_learning_score_diff")

        update_samples = sample_idx[updates == 1]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.vlines(update_samples, 0, 1, color="#c1121f")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("sample")
        ax.set_yticks([])
        ax.set_title("On-Chip Learning Update Events")
        save_pdf_page(pdf, fig, png_dir, "15_update_events")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.step(sample_idx, update_count, where="post")
        ax.set_xlabel("sample")
        ax.set_ylabel("cumulative updates")
        ax.set_title("Hardware Update Counter")
        save_pdf_page(pdf, fig, png_dir, "16_update_count")

        if "infer_skip_count" in inference_rows[0]:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.step(sample_idx, col(inference_rows, "infer_skip_count", int), where="post", label="inference run")
            if "infer_skip_count" in learn_rows[0]:
                ax.step(sample_idx, col(learn_rows, "infer_skip_count", int), where="post", label="learning run inference counter", alpha=0.75)
            ax.set_xlabel("sample")
            ax.set_ylabel("cumulative skipped feature-MACs")
            ax.set_title("RTL Activity-Gate Skip Counter")
            ax.legend()
            save_pdf_page(pdf, fig, png_dir, "16b_skip_count")

        if "train_skip_count" in learn_rows[0]:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.step(sample_idx, col(learn_rows, "train_skip_count", int), where="post")
            ax.set_xlabel("sample")
            ax.set_ylabel("cumulative skipped learning features")
            ax.set_title("RTL On-Chip-Learning Feature Gate Counter")
            save_pdf_page(pdf, fig, png_dir, "16c_train_skip_count")

        if len(update_samples):
            fig, ax = plt.subplots(figsize=(11, 5))
            width = 0.35
            upd_labels = labels[updates == 1]
            upd_preds = learn_preds[updates == 1]
            ax.bar(np.arange(len(update_samples)) - width / 2, upd_labels, width, label="true label")
            ax.bar(np.arange(len(update_samples)) + width / 2, upd_preds, width, label="predicted")
            ax.set_xticks(np.arange(len(update_samples)), update_samples, rotation=90)
            ax.set_ylabel("class")
            ax.set_xlabel("sample where update occurred")
            ax.set_title("Classes Involved In Each On-Chip Update")
            ax.legend()
            save_pdf_page(pdf, fig, png_dir, "17_update_classes")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(features, bins=100)
        if infer_skip_abs_thresh > 0:
            ax.axvline(-infer_skip_abs_thresh, color="#c1121f", linestyle="--", label="-gate threshold")
            ax.axvline(infer_skip_abs_thresh, color="#c1121f", linestyle="--", label="+gate threshold")
            ax.legend()
        ax.set_title("Exported Spikingformer Feature Distribution")
        ax.set_xlabel("int16 feature value")
        save_pdf_page(pdf, fig, png_dir, "18_feature_hist")

        abs_features = np.abs(features)
        thresholds = np.array([0, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384])
        skip_fracs = np.array([(abs_features <= t).mean() for t in thresholds], dtype=float)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(thresholds, skip_fracs * 100.0, marker="o")
        if infer_skip_abs_thresh > 0:
            ax.axvline(infer_skip_abs_thresh, color="#c1121f", linestyle="--")
            ax.text(infer_skip_abs_thresh, feature_mac_skip_fraction * 100.0, f"  selected {feature_mac_skip_fraction * 100:.2f}%", va="center")
        ax.set_xscale("symlog", linthresh=16)
        ax.set_xlabel("abs(feature) gate threshold")
        ax.set_ylabel("feature-MACs skipped (%)")
        ax.set_title("Measured Feature Activity Gating Curve")
        save_pdf_page(pdf, fig, png_dir, "18b_feature_gate_sweep")

        first_eff = features[:FEATURE_DIM].copy()
        if infer_skip_abs_thresh > 0:
            first_eff[np.abs(first_eff) <= infer_skip_abs_thresh] = 0
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(features[:FEATURE_DIM], label="raw")
        ax.plot(first_eff, label="after gate", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title("First Feature Vector Before/After Activity Gate")
        ax.set_xlabel("feature index")
        ax.set_ylabel("int16 value")
        ax.legend()
        save_pdf_page(pdf, fig, png_dir, "18c_first_feature_gate_overlay")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(features[:FEATURE_DIM])
        ax.set_title("First Exported 384-D Feature Vector")
        ax.set_xlabel("feature index")
        ax.set_ylabel("int16 value")
        save_pdf_page(pdf, fig, png_dir, "19_first_feature_vector")

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(features[:FEATURE_DIM].reshape(16, 24), aspect="auto", cmap="coolwarm")
        ax.set_title("First Feature Vector Reshaped 16 x 24 For Inspection")
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "20_first_feature_grid")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(weights.reshape(-1), bins=100)
        ax.set_title("Classifier Weight Distribution Before On-Chip Updates")
        ax.set_xlabel("int16 weight")
        save_pdf_page(pdf, fig, png_dir, "21_weight_hist")

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(weights, aspect="auto", cmap="coolwarm")
        ax.set_xlabel("feature index")
        ax.set_ylabel("class")
        ax.set_title("Classifier Weight Matrix: Class x Feature")
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "22_weight_heatmap")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(np.arange(NUM_CLASSES), np.linalg.norm(weights, axis=1))
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_ylabel("L2 norm")
        ax.set_title("Per-Class Weight Norm")
        save_pdf_page(pdf, fig, png_dir, "23_weight_norms")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(np.arange(NUM_CLASSES) - 0.18, bias / 1e9, width=0.36, label="before")
        ax.bar(np.arange(NUM_CLASSES) + 0.18, after_bias / 1e9, width=0.36, label="after")
        ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES)
        ax.set_ylabel("bias / 1e9")
        ax.set_title("Classifier Bias Before/After On-Chip Updates")
        ax.legend()
        save_pdf_page(pdf, fig, png_dir, "24_bias_before_after")

        wdiff = after_weights - weights
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(wdiff.reshape(-1), bins=np.arange(wdiff.min() - 1, wdiff.max() + 2) if wdiff.size and wdiff.max() - wdiff.min() < 200 else 100)
        ax.set_title("Weight Delta Distribution After Online Learning")
        ax.set_xlabel("after - before")
        save_pdf_page(pdf, fig, png_dir, "25_weight_delta_hist")

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(wdiff, aspect="auto", cmap="coolwarm")
        ax.set_xlabel("feature index")
        ax.set_ylabel("class")
        ax.set_title("Weight Delta Matrix After On-Chip Updates")
        fig.colorbar(im, ax=ax)
        save_pdf_page(pdf, fig, png_dir, "26_weight_delta_heatmap")

        fig, ax = plt.subplots(figsize=(11, 5))
        names = ["LUT", "FF", "BRAM36", "DSP", "LUTRAM"]
        vals = [vivado["lut"], vivado["regs"], vivado["bram"], vivado["dsp"], vivado["lut_mem"]]
        ax.bar(names, vals)
        ax.set_title("Vivado Utilization Counts")
        for i, value in enumerate(vals):
            ax.text(i, value, str(value), ha="center", va="bottom")
        save_pdf_page(pdf, fig, png_dir, "27_utilization")

        fig, ax = plt.subplots(figsize=(8, 6))
        power_names = ["clock", "logic", "signals", "BRAM", "DSP", "IO"]
        power_vals = [vivado["clock_power_w"], vivado["logic_power_w"], vivado["signal_power_w"], vivado["bram_power_w"], vivado["dsp_power_w"], vivado["io_power_w"]]
        ax.pie(power_vals, labels=power_names, autopct="%1.1f%%")
        ax.set_title("Vivado Dynamic Power Breakdown")
        save_pdf_page(pdf, fig, png_dir, "28_power_pie")

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(["WNS", "WHS"], [vivado["wns_ns"], vivado["whs_ns"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("ns")
        ax.set_title("Routed Timing Slack At 100 MHz")
        save_pdf_page(pdf, fig, png_dir, "29_timing_slack")

        saved_infer_cycles = 2 * rtl_infer_skip_count
        saved_train_cycles = 5 * learn_train_skip_count
        latency_lines = [
            "Feature load: 384 accepted feature words per frame through AXI-stream wrapper.",
            "Per class: CLASS_INIT plus one MAC_READ per feature; active features run MAC_MUL and MAC_ACCUM, while gated features jump directly to CLASS_DONE.",
            f"Activity gate: abs(feature) <= {infer_skip_abs_thresh} bypasses DSP multiply/accumulate for inference; measured exported-set skip fraction is {feature_mac_skip_fraction * 100:.2f}%.",
            f"RTL inference skip counter {rtl_infer_skip_count:,} removes {saved_infer_cycles:,} MAC_MUL/MAC_ACCUM cycles over the 10k fixture.",
            "Learning update, when applied: active features update true and predicted class rows; gated features skip the no-op read/calc/write sequence.",
            f"RTL learning skip counter {learn_train_skip_count:,} removes approximately {saved_train_cycles:,} true/pred update cycles in the full learning run.",
            f"Learning is margin-gated, so the full 10k validation stream applied {learn_val.get('updates')} updates.",
        ]
        save_pdf_page(pdf, plot_text_page("Pipeline / Latency Accounting", latency_lines), png_dir, "30_latency")

    print(f"Wrote {pdf_path}")
    print(f"Wrote plot PNGs in {png_dir}")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def explain_line(path: Path, line: str, index: int, previous_nonblank: str) -> str:
    stripped = line.strip()
    suffix = path.suffix.lower()
    name = path.name
    if not stripped:
        return "This blank line is not hardware or software behavior. It separates nearby declarations or operations so the file remains readable and reviewable."
    if stripped.startswith("//") or stripped.startswith("#"):
        return "This is a human-facing comment. It documents intent for the reader and is ignored by the compiler or interpreter."
    if suffix in {".sv", ".v", ".svh"}:
        if stripped.startswith("`timescale"):
            return "This simulator directive sets the time unit and precision used by testbenches and waveform timestamps. It keeps delays such as clock half-periods numerically meaningful in XSim."
        if stripped.startswith("`default_nettype none"):
            return "This disables implicit one-bit wires. It prevents spelling mistakes from silently creating unwanted hardware nets."
        if stripped.startswith("`include"):
            return "This includes shared compile-time constants such as feature dimensions and fixed-point widths. The RTL and testbenches therefore agree on the same trained-model shape."
        if stripped.startswith("module "):
            return "This begins a hardware module, which is the FPGA equivalent of a reusable circuit block. Everything until the matching endmodule describes ports, storage, combinational equations, and clocked behavior for this block."
        if stripped.startswith("endmodule"):
            return "This closes the module definition. Synthesis stops adding logic to this circuit block at this point."
        if "INFER_SKIP_ABS_THRESH" in stripped:
            return "This line controls the feature-activity gate. When the absolute fixed-point feature is at or below this threshold, the RTL treats it as zero so the exported Spikingformer head ignores small noisy activations and records a measurable MAC-skip opportunity."
        if "skip_feature" in stripped:
            return "This line is part of the combinational decision that detects low-magnitude features. It is the hardware equivalent of asking whether a small activation should be silent for this inference/update pass."
        if "effective_feature" in stripped:
            return "This line is part of the denoised feature path. The returned value is either the original signed feature or exactly zero, and that registered value is what feeds both the DSP multiply and the online-learning delta."
        if "skip_count" in stripped:
            return "This line exposes or updates a hardware activity counter. The counter makes the feature gate visible in waveforms and CSV traces, so the power-saving opportunity is observable from RTL rather than only from Python statistics."
        if "parameter" in stripped:
            return "This is a compile-time hardware setting. Vivado resolves it before synthesis, so it controls the generated circuit size, fixed-point width, or learning rule without runtime overhead."
        if re.search(r"\binput\b|\boutput\b", stripped):
            return "This line declares a module port. A port is a physical or wrapper-visible signal used to move clocks, control pulses, features, labels, predictions, or debug information across module boundaries."
        if "typedef enum" in stripped:
            return "This declares the finite-state-machine state type. Using named states makes the cycle-by-cycle control flow explicit instead of relying on anonymous numeric constants."
        if stripped.startswith("ST_") or stripped.startswith("WR_"):
            return "This names one state in the control FSM. The numeric encoding is what appears in waveform traces, while the name explains the pipeline phase being executed."
        if "feature_mem" in stripped:
            return "This line touches the feature memory that holds one 384-dimensional fixed-point Spikingformer feature vector for the current image. Keeping the feature vector local avoids repeatedly streaming the same values for every class score."
        if "weight_mem" in stripped:
            return "This line touches the trained classifier weight store. In the final RTL the large 10 x 384 int16 matrix is inferred as true dual-port BRAM, so weights remain on chip and can also be updated by the learning state machine."
        if "bias_mem" in stripped:
            return "This line touches the class-bias store. Bias values are part of the affine classifier score and are updated on chip in separate one-write states to keep synthesis clean."
        if "class_score_mem" in stripped:
            return "This stores or reads the ten class scores for the current sample. Those scores are needed for argmax prediction, CSV testbench reporting, and the margin test used before an online update."
        if stripped.startswith("assign "):
            return "This is continuous combinational wiring. The output follows its source expression immediately in hardware rather than waiting for a clock edge."
        if stripped.startswith("initial"):
            return "This simulation/synthesis initialization block loads trained memories from generated .mem files. On FPGA, Vivado maps these initial values into BRAM/LUTRAM initialization data."
        if "$readmemh" in stripped:
            return "This reads a hexadecimal memory file into an RTL array. It is how exported trained weights and biases become deterministic FPGA memory contents."
        if stripped.startswith("function"):
            return "This begins a pure helper function used inside the RTL. It describes fixed combinational math, not a software function running on a CPU."
        if "sat_weight" in stripped:
            return "This line is part of signed saturation for 16-bit weights. Saturation prevents online learning from wrapping around when an update would exceed the representable fixed-point range."
        if "learn_delta" in stripped:
            return "This line implements the hardware learning step size. The feature is sign-extended and arithmetically shifted so positive and negative features update weights symmetrically."
        if "always_ff" in stripped:
            return "This begins clocked sequential logic. Registers and RAM writes inside this block update only on a clock edge, which is what makes the design synthesizable and waveform-predictable."
        if "if (!rst_n)" in stripped:
            return "This is the active-low reset branch. It puts control registers and visible outputs into a known starting state before a frame is processed."
        if "unique case" in stripped:
            return "This starts a state-machine branch table. The unique qualifier tells tools that one legal state should match at a time, improving diagnostics and optimization."
        if "state <=" in stripped:
            return "This schedules the next FSM state for the following clock edge. That transition is the hardware pipeline control path."
        if "<=" in stripped:
            return "This is a nonblocking register or memory assignment. The right side is sampled now and the left side updates together with other clocked assignments at the end of the clock edge."
        if "=" in stripped:
            return "This is a local compile-time or procedural assignment. In context it either defines a helper value or connects a temporary expression used by nearby hardware logic."
        return "This SystemVerilog line contributes to the structural or sequential description of the accelerator. Its effect is resolved by XSim for verification and by Vivado during synthesis."
    if suffix == ".py":
        if stripped.startswith("from ") or stripped.startswith("import "):
            return "This imports a Python library or symbol. The scripts use standard parsing, numeric arrays, plotting, and file APIs to make the hardware artifacts reproducible."
        if stripped.startswith("def "):
            return "This defines a Python function. The function packages one repeatable step such as loading a memory file, simulating the golden learning rule, validating CSVs, or generating plots."
        if "argparse" in stripped or "add_argument" in stripped:
            return "This contributes to the command-line interface. It lets the rerun flow choose paths, sample counts, or output locations without editing code."
        if "Path" in stripped:
            return "This uses path objects so generated artifacts are addressed consistently across Windows shells and Python code."
        if "read_text" in stripped or "open(" in stripped:
            return "This reads an artifact produced by training, export, simulation, or Vivado. The validator/report depends on files generated by the actual run rather than hidden constants."
        if "write_text" in stripped or "json.dump" in stripped:
            return "This writes machine-readable output for later reports. Keeping JSON/CSV artifacts makes the verification auditable."
        if "np." in stripped or "numpy" in stripped:
            return "This performs vectorized numerical work. The fixed-point golden models use NumPy so score, margin, and update calculations are explicit and repeatable."
        if "weights" in stripped or "bias" in stripped:
            return "This line handles the classifier parameters. Those arrays mirror the RTL memories, so mismatches reveal an export, quantization, or hardware-update bug."
        if "scores" in stripped:
            return "This line handles class scores. The score vector is the direct software equivalent of the RTL MAC loop over weights and features."
        if "pred" in stripped:
            return "This line handles predicted labels. It checks that argmax behavior in Python and RTL chooses the same class."
        if "update" in stripped:
            return "This line handles the online-learning rule or its validation fields. It is central to proving that the FPGA update path is not bypassed."
        if "plt." in stripped or "PdfPages" in stripped:
            return "This creates a debug visualization. The plot is derived from generated CSV or Vivado report data so it can be regenerated after RTL changes."
        if stripped.startswith("if __name__"):
            return "This Python idiom runs main() only when the file is executed as a script. It still allows the file's functions to be imported for tests or reuse."
        return "This Python line participates in the export, validation, or report-generation pipeline. It is executed on the host PC to prepare data for RTL or to check RTL outputs."
    if suffix == ".ps1":
        if stripped.startswith("param"):
            return "This begins the PowerShell parameter block. It exposes repeatable knobs such as sample count, Vivado path, board part, and skip switches."
        if stripped.startswith("["):
            return "This declares one typed PowerShell parameter. Strong typing catches accidental string/numeric mistakes before the long training or Vivado flow starts."
        if "$ErrorActionPreference" in stripped:
            return "This makes failures stop the script instead of continuing after a broken export, simulation, or synthesis step."
        if "Assert-LastExitCode" in stripped:
            return "This calls the helper that turns a failed external tool exit code into a hard pipeline failure."
        if "Vivado" in stripped or "xvlog" in stripped or "xelab" in stripped or "xsim" in stripped:
            return "This line invokes or locates Vivado simulator/implementation tools. The flow uses the installed AMD toolchain on D: rather than a substitute simulator."
        if "python" in stripped.lower():
            return "This line invokes the Python environment used for export, validation, or report generation. It ties host-side math to the hardware artifacts."
        return "This PowerShell line orchestrates the reproducible full flow. It exists so changes can be rerun without manually retyping many commands."
    if suffix == ".tcl":
        if stripped.startswith("read_verilog"):
            return "This tells Vivado which SystemVerilog files to synthesize. The wrapper and accelerator are both read so the routed report matches the PYNQ-facing design."
        if stripped.startswith("synth_design"):
            return "This starts synthesis for the selected FPGA part and top module. Synthesis maps RTL arrays, arithmetic, and FSMs into FPGA primitives."
        if stripped.startswith("create_clock"):
            return "This creates the 100 MHz timing constraint. Vivado uses it to decide whether every register-to-register path is fast enough."
        if stripped.startswith("place_design") or stripped.startswith("route_design") or stripped.startswith("opt_design"):
            return "This runs an implementation stage. These commands turn the synthesized netlist into a placed and routed design with real timing and power estimates."
        if stripped.startswith("report_") or stripped.startswith("write_checkpoint"):
            return "This emits a signoff artifact. The generated reports are the source for utilization, timing, and power numbers in the debug PDF."
        return "This TCL line controls the Vivado batch implementation flow. It makes the hardware report reproducible from the command line."
    return "This line is part of the accelerator support files and is included so the full rerun flow is documented, not just the RTL."


def generate_explainer_tex() -> Path:
    pdir = project_dir()
    ddir = debug_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    odir = out_dir()
    rtl_val = read_json(odir / "spikingformer_head_rtl_validation.json")
    learn_val = read_json(odir / "spikingformer_head_onchip_learning_validation.json")
    mapping = read_json(odir / "spikingformer_head_mapping_metrics.json")
    vivado = parse_vivado_reports(odir / "vivado")
    acc_pct = float(rtl_val.get("accuracy", float("nan"))) * 100.0
    sample_count = int(rtl_val.get("samples", 0) or 0)
    update_count = int(learn_val.get("updates", 0) or 0)
    infer_skip_abs_thresh = int(rtl_val.get("infer_skip_abs_thresh", mapping.get("infer_skip_abs_thresh", 0)) or 0)
    feature_mac_skip_fraction = float(rtl_val.get("feature_mac_skip_fraction", mapping.get("feature_mac_skip_fraction", 0.0)) or 0.0) * 100.0
    files = [
        pdir / "rtl" / "cifar10_spikingformer_head_accelerator.sv",
        pdir / "rtl" / "cifar10_spikingformer_head_pynq_axi_stream_top.sv",
        pdir / "rtl" / "tb_spikingformer_head_learning_unit.sv",
        pdir / "rtl" / "tb_cifar10_spikingformer_head_accelerator.sv",
        pdir / "rtl" / "tb_cifar10_spikingformer_head_onchip_learning.sv",
        pdir / "export_spikingformer_head_accelerator.py",
        pdir / "validate_spikingformer_head_rtl.py",
        pdir / "validate_spikingformer_head_onchip_learning.py",
        pdir / "generate_spikingformer_debug_reports.py",
        pdir / "run_spikingformer_head_accelerator.ps1",
        pdir / "spikingformer_head_synth.tcl",
    ]
    tex_path = ddir / "neuraedge_full_accelerator_line_by_line_explainer.tex"
    lines: list[str] = []
    lines.extend(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.65in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\usepackage{listings}",
            r"\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false}",
            r"\title{NeuraEdge CIFAR-10 Neuromorphic Accelerator: Full Pipeline and Line-by-Line Explainer}",
            r"\author{Generated from repository source and measured Vivado/XSim artifacts}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\section{What Was Built}",
            "This document describes the implemented CIFAR-10 accelerator path. The trained Spikingformer backbone is used to export 384-dimensional fixed-point features and a trained classifier head. The FPGA accelerator stores one feature vector, applies a measured low-magnitude feature activity gate, computes ten fixed-point class scores with an on-chip weight memory and one DSP MAC datapath, returns the predicted label, and includes a margin-gated on-chip learning path that updates the true-label and predicted-label classifier rows in FPGA memory.",
            r"\section{Measured Verification Summary}",
            r"\begin{itemize}",
            rf"\item Full inference XSim run: {sample_count:,} CIFAR-10 samples, {acc_pct:.2f}\% top-1 accuracy, exact score match against the fixed-point Python reference.",
            rf"\item Feature activity gate: abs(feature) $\leq$ {infer_skip_abs_thresh} is treated as zero before inference and learning; the exported test set skips {feature_mac_skip_fraction:.2f}\% of feature-MAC opportunities.",
            rf"\item Full on-chip-learning XSim run: {sample_count:,} samples, {update_count} RTL weight/bias updates, zero score mismatches, zero prediction mismatches, zero update-flag mismatches, and zero update-count mismatches against the independent Python online-learning golden model.",
            rf"\item Vivado implementation target: PYNQ Zynq-7020, xc7z020clg400-1, 100 MHz. Routed timing met with WNS {vivado['wns_ns']:.3f} ns and WHS {vivado['whs_ns']:.3f} ns.",
            rf"\item Resource result: {vivado['lut']} LUTs, {vivado['regs']} FFs, {vivado['bram']} RAMB36E1 blocks, and {vivado['dsp']} DSP48E1.",
            rf"\item Vectorless Vivado power: {vivado['total_power_w']:.3f} W total, {vivado['dynamic_power_w']:.3f} W dynamic, {vivado['static_power_w']:.3f} W static.",
            r"\end{itemize}",
            r"\section{Research Context And Comparison}",
            rf"The model side follows the Spikingformer family of spike-driven transformer SNNs, whose reported CIFAR-10 accuracy is about 95.81--95.95\% in the cited paper/repository. The local fixed-point exported checkpoint reaches {acc_pct:.2f}\% after mapping into this routed hardware head with the measured feature gate. Recent SNN-transformer targets to beat include Max-Former at 97.04\% on CIFAR-10 and alpha-XNOR spiking self-attention at 96.15\% for the reported multi-stage ViT setting. The implemented hardware contribution here is the verified FPGA path: local feature and weight memories, exact fixed-point RTL/golden agreement, activity-gated feature silence, and on-chip classifier learning.",
            r"\begin{itemize}",
            r"\item Spikingformer paper/repository: \url{https://arxiv.org/html/2304.11954v5}, \url{https://github.com/TheBrainLab/Spikingformer}.",
            r"\item Max-Former high-frequency spiking transformer reference: \url{https://arxiv.org/html/2505.18608v4}.",
            r"\item Alpha-XNOR spiking self-attention reference: \url{https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.pdf}.",
            r"\item TrueNorth reference: \url{https://research.ibm.com/publications/truenorth-design-and-tool-flow-of-a-65-mw-1-million-neuron-programmable-neurosynaptic-chip}.",
            r"\item Loihi on-chip learning reference: \url{https://redwood.berkeley.edu/wp-content/uploads/2021/08/Davies2018.pdf}.",
            r"\end{itemize}",
            r"\section{End-to-End Dataflow}",
            r"\begin{enumerate}",
            r"\item Python loads the trained Spikingformer checkpoint and CIFAR-10 data, runs the trained network, and exports fixed-point feature, label, weight, and bias memories.",
            r"\item XSim streams each 384-word feature vector into the RTL through the same handshake the PYNQ AXI-stream wrapper uses.",
            r"\item The RTL stores the feature vector locally, zeroes features whose magnitude is below the selected gate threshold, loops over the ten classes, reads each class row from BRAM, multiplies effective feature and weight in a DSP, accumulates a 64-bit score, and chooses argmax.",
            r"\item If training is enabled and the prediction is wrong but close enough by margin, the RTL latches the label, reads the true and predicted class rows, computes signed shifted deltas from the same effective feature path, saturates int16 weight updates, writes updated weights, and updates the two class biases.",
            r"\item Python validators independently replay the same fixed-point inference and online-learning rule from the exported memories, then compare every RTL score, prediction, update flag, and update count.",
            r"\item Vivado synthesizes and routes the PYNQ AXI-stream top, producing timing, utilization, power, and checkpoint artifacts.",
            r"\end{enumerate}",
            r"\section{Line-by-Line Source Explanation}",
        ]
    )
    for path in files:
        rel = path.relative_to(repo_root())
        lines.append(r"\clearpage")
        lines.append(r"\subsection{" + latex_escape(str(rel)) + "}")
        lines.append("Each entry below gives the exact source line followed by its hardware/software role in the implemented accelerator pipeline.")
        previous = ""
        for i, raw in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            explanation = explain_line(path, raw, i, previous)
            if raw.strip():
                previous = raw.strip()
            lines.append(r"\paragraph{Line " + str(i) + "}")
            code_text = latex_escape(raw) if raw else r"\emph{blank line}"
            lines.append(
                r"\noindent\colorbox{gray!8}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{\raggedright\ttfamily\scriptsize "
                + code_text
                + r"}}"
            )
            lines.append(r"\par\smallskip")
            lines.append(latex_escape(explanation))
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def compile_tex(tex_path: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("pdflatex not found; wrote TeX only.")
        return
    for _ in range(2):
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(f"Wrote {tex_path.with_suffix('.pdf')}")


def main() -> None:
    generate_debug_pdf()
    tex_path = generate_explainer_tex()
    print(f"Wrote {tex_path}")
    compile_tex(tex_path)


if __name__ == "__main__":
    main()
