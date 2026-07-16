"""Generate model, compiler, waveform, timing, utilization, and power plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
COLORS = ["#155e75", "#166534", "#991b1b", "#7c3e00", "#4f46e5", "#374151"]


@dataclass
class PlotRecord:
    filename: str
    title: str
    evidence: str


@dataclass
class VcdSignal:
    name: str
    width: int
    transitions: list[tuple[int, float]]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def style_axes(axis: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, fontsize=12, fontweight="bold", loc="left")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    pdf: PdfPages,
    records: list[PlotRecord],
    index: int,
    slug: str,
    title: str,
    evidence: str,
) -> int:
    figure.tight_layout()
    filename = f"{index:02d}_{slug}.png"
    figure.savefig(output_dir / filename, dpi=190, bbox_inches="tight")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    records.append(PlotRecord(filename, title, evidence))
    return index + 1


def text_page(title: str, lines: Sequence[str]) -> plt.Figure:
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.patch.set_facecolor("white")
    figure.text(0.055, 0.93, title, fontsize=20, fontweight="bold", color="#111827")
    y = 0.86
    for line in lines:
        wrapped = [line[start : start + 125] for start in range(0, len(line), 125)] or [""]
        for segment in wrapped:
            figure.text(0.065, y, segment, fontsize=10.5, family="monospace", color="#1f2937")
            y -= 0.038
        y -= 0.008
    return figure


def confusion_figure(matrix: np.ndarray, title: str, signed: bool = False) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9.5, 8.2))
    if signed:
        limit = max(1, int(np.abs(matrix).max()))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(10), CLASSES)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title, fontsize=14, fontweight="bold", loc="left")
    threshold = np.max(np.abs(matrix)) * 0.55
    for row in range(10):
        for column in range(10):
            value = int(matrix[row, column])
            if value != 0 or not signed:
                axis.text(column, row, str(value), ha="center", va="center", fontsize=8,
                          color="white" if abs(value) > threshold else "black")
    return figure


def parse_vcd(path: Path) -> tuple[str, dict[str, VcdSignal]]:
    scopes: list[str] = []
    code_definitions: dict[str, list[tuple[str, int]]] = defaultdict(list)
    timescale = "1ps"
    in_definitions = True
    selected_codes: set[str] = set()
    transitions: dict[str, list[tuple[int, float]]] = defaultdict(list)
    current_time = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if in_definitions:
                if line.startswith("$timescale"):
                    pieces = line.split()
                    if len(pieces) >= 2 and pieces[1] != "$end":
                        timescale = pieces[1]
                elif line.startswith("$scope"):
                    pieces = line.split()
                    scopes.append(pieces[2])
                elif line.startswith("$upscope"):
                    if scopes:
                        scopes.pop()
                elif line.startswith("$var"):
                    pieces = line.split()
                    if len(pieces) >= 6:
                        width = int(pieces[2])
                        code = pieces[3]
                        reference = pieces[4]
                        name = "/" + "/".join([*scopes, reference])
                        code_definitions[code].append((name, width))
                elif line.startswith("$enddefinitions"):
                    in_definitions = False
                    selected_codes = set(code_definitions)
                continue
            if not line or line.startswith("$"):
                continue
            if line.startswith("#"):
                current_time = int(line[1:])
                continue
            if line[0] in "01xXzZ":
                code = line[1:]
                if code in selected_codes:
                    value = float(line[0]) if line[0] in "01" else math.nan
                    transitions[code].append((current_time, value))
            elif line[0] in "bBrR":
                pieces = line.split(maxsplit=1)
                if len(pieces) != 2:
                    continue
                encoded, code = pieces
                if code not in selected_codes:
                    continue
                digits = encoded[1:]
                if any(character.lower() in "xz" for character in digits):
                    value = math.nan
                elif encoded[0].lower() == "b":
                    value = float(int(digits, 2))
                else:
                    try:
                        value = float(digits)
                    except ValueError:
                        value = math.nan
                transitions[code].append((current_time, value))

    signals: dict[str, VcdSignal] = {}
    for code, definitions in code_definitions.items():
        values = transitions.get(code, [])
        for name, width in definitions:
            signals[name] = VcdSignal(name, width, values)
    return timescale, signals


def choose_wave_signals(signals: dict[str, VcdSignal], limit: int = 10) -> list[VcdSignal]:
    priority = (
        "program_active", "program_done", "program_error", "state", "command_valid",
        "input_fifo_count", "s_axis_tvalid", "s_axis_tready", "core_s_valid",
        "core_s_ready", "issue_valid", "issue_ready", "result_valid", "result_ready",
        "output_valid", "output_ready", "spike", "membrane", "dense_macs", "spike_adds",
        "update_count", "result_fifo_count", "m_axis_tlast", "interrupt", "product",
        "activation", "weight",
    )
    candidates: list[tuple[int, int, str, VcdSignal]] = []
    for signal in signals.values():
        lowered = signal.name.lower()
        leaf = lowered.rsplit("/", 1)[-1]
        if lowered.endswith("/clk") or not signal.transitions:
            continue
        if leaf in {"clear_state", "lif_clear_state", "clear_counters"}:
            continue
        matches = [
            index
            for index, token in enumerate(priority)
            if leaf == token or leaf == token + "_r" or (token != "state" and token in leaf)
        ]
        if not matches:
            continue
        # Prefer DUT registers and shallow hierarchy so plots remain readable.
        hierarchy_penalty = signal.name.count("/")
        dut_penalty = 0 if "/dut/" in lowered else 2
        candidates.append((min(matches), hierarchy_penalty + dut_penalty, signal.name, signal))
    chosen: list[VcdSignal] = []
    used_leaf_names: set[str] = set()
    for _, _, _, signal in sorted(candidates):
        leaf = signal.name.rsplit("/", 1)[-1]
        if leaf in used_leaf_names:
            continue
        used_leaf_names.add(leaf)
        chosen.append(signal)
        if len(chosen) == limit:
            break
    return chosen


def waveform_figure(vcd_path: Path, title: str) -> tuple[plt.Figure, list[str]]:
    timescale, signals = parse_vcd(vcd_path)
    selected = choose_wave_signals(signals)
    if not selected:
        raise RuntimeError(f"no debug signals selected from {vcd_path}")
    figure, axes = plt.subplots(len(selected), 1, figsize=(12.5, 1.25 * len(selected) + 1.2), sharex=True)
    axes_array = np.atleast_1d(axes)
    unit_scale = {"1ps": 1e-3, "1ns": 1.0}.get(timescale, 1.0)
    for axis, signal, color in zip(axes_array, selected, COLORS * 3):
        data = signal.transitions
        if len(data) > 2500:
            stride = math.ceil(len(data) / 2500)
            data = data[::stride]
        x = np.asarray([item[0] * unit_scale for item in data], dtype=np.float64)
        y = np.asarray([item[1] for item in data], dtype=np.float64)
        if signal.width == 1:
            axis.step(x, y, where="post", color=color, linewidth=1.2)
            axis.set_ylim(-0.2, 1.2)
            axis.set_yticks([0, 1])
        else:
            axis.step(x, y, where="post", color=color, linewidth=1.05)
        axis.set_ylabel(signal.name.rsplit("/", 1)[-1], rotation=0, ha="right", va="center", fontsize=8)
        axis.grid(True, axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes_array[0].set_title(title, fontsize=13, fontweight="bold", loc="left")
    axes_array[-1].set_xlabel(f"Simulation time ({'ns' if unit_scale == 1 else 'ns; VCD timescale ' + timescale})")
    return figure, [signal.name for signal in selected]


def arithmetic_groups(operators: Sequence[dict]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for operator in operators:
        matrix = operator.get("matrix")
        if matrix:
            totals[matrix["arithmetic"]] += int(matrix["mac_equivalents"])
    return dict(totals)


def parse_timing_slacks(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    values = [float(value) for value in re.findall(r"Slack \((?:MET|VIOLATED)\)\s*:\s*(-?\d+\.\d+)ns", text)]
    return np.asarray(values, dtype=np.float64)


def parse_implementation_summary(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in raw:
            continue
        key, value = [part.strip() for part in raw.split("=", 1)]
        try:
            result[key] = float(value)
        except ValueError:
            continue
    return result


def parse_clock_timing(path: Path) -> dict[str, tuple[float, float]]:
    """Return routed setup/hold slack for each FPGA clock-domain row."""
    clocks: dict[str, tuple[float, float]] = {}
    if not path.exists():
        return clocks
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = raw.split()
        if len(fields) < 10 or not re.fullmatch(r"clk_fpga_\d+", fields[0]):
            continue
        try:
            setup_slack = float(fields[1])
            hold_slack = float(fields[5])
        except ValueError:
            continue
        clocks.setdefault(fields[0], (setup_slack, hold_slack))
    return clocks


def parse_utilization_hierarchy(path: Path) -> dict[str, dict[str, int]]:
    """Parse named rows from Vivado's hierarchical utilization table."""
    wanted = {
        "system_wrapper", "accelerator", "accelerator_control_cdc",
        "control_interconnect", "dma", "dma_control_cdc",
        "memory_interconnect", "mm2s_cdc", "s2mm_cdc",
    }
    rows: dict[str, dict[str, int]] = {}
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw.startswith("|"):
            continue
        fields = [field.strip() for field in raw.split("|")]
        if len(fields) < 12 or fields[1] not in wanted:
            continue
        try:
            rows[fields[1]] = {
                "lut": int(fields[3]), "ff": int(fields[7]),
                "bram36": int(fields[8]), "bram18": int(fields[9]),
                "dsp": int(fields[10]),
            }
        except ValueError:
            continue
    return rows


def parse_drc_summary(path: Path) -> list[tuple[str, str, str, int]]:
    """Parse DRC rule, severity, description, and count from the summary table."""
    if not path.exists():
        return []
    pattern = re.compile(
        r"^\|\s*([A-Z]+-\d+)\s*\|\s*(\w+)\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|$"
    )
    rows: list[tuple[str, str, str, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(raw)
        if match:
            rows.append((match.group(1), match.group(2), match.group(3), int(match.group(4))))
    return rows


def parse_cdc_summary(path: Path) -> list[tuple[str, str, int, str]]:
    """Parse the compact CDC classification table before detailed paths."""
    if not path.exists():
        return []
    pattern = re.compile(r"^(CDC-\d+)\s+(\w+)\s+(\d+)\s+(.+)$")
    rows: list[tuple[str, str, int, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(raw.strip())
        if match:
            rows.append((match.group(1), match.group(2), int(match.group(3)), match.group(4)))
        elif rows:
            break
    return rows


def parse_power_report(path: Path) -> dict[str, object]:
    """Extract explicit Vivado power totals, resources, hierarchy, and confidence."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")

    def number(label: str) -> float | None:
        match = re.search(rf"^\|\s*{re.escape(label)}\s*\|\s*([0-9.]+)", text, re.MULTILINE)
        return float(match.group(1)) if match else None

    confidence_match = re.search(
        r"^\|\s*Confidence Level\s*\|\s*(\w+)", text, re.MULTILINE
    )
    matched_match = re.search(
        r"^\|\s*Design Nets Matched\s*\|\s*([^|]+)", text, re.MULTILINE
    )
    resources: dict[str, float] = {}
    for label in ("Clocks", "Slice Logic", "Signals", "Block RAM", "DSPs", "PS7", "Static Power", "Total"):
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*<?([0-9.]+)", text, re.MULTILINE
        )
        if match:
            resources[label] = float(match.group(1))
    hierarchy: dict[str, float] = {}
    hierarchy_names = (
        "system_wrapper", "system_i", "accelerator", "accelerator_control_cdc",
        "control_interconnect", "dma", "dma_control_cdc", "memory_interconnect",
        "mm2s_cdc", "ps7", "reset_100", "reset_125", "reset_200", "s2mm_cdc",
    )
    for name in hierarchy_names:
        match = re.search(
            rf"^\|\s*{re.escape(name)}\s*\|\s*([0-9.]+)\s*\|", text, re.MULTILINE
        )
        if match:
            hierarchy[name] = float(match.group(1))
    return {
        "total": number("Total On-Chip Power (W)"),
        "dynamic": number("Dynamic (W)"),
        "static": number("Device Static (W)"),
        "confidence": confidence_match.group(1) if confidence_match else "unknown",
        "matched": matched_match.group(1).strip() if matched_match else "not reported",
        "resources": resources,
        "hierarchy": hierarchy,
    }


def file_evidence(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def latex_breakable(value: str) -> str:
    """Escape provenance text while adding legal line-break opportunities."""
    escaped = latex_escape(value)
    escaped = escaped.replace(r"\textbackslash{}", r"\textbackslash{}\allowbreak{}")
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    escaped = escaped.replace("/", r"/\allowbreak{}")
    escaped = escaped.replace(";", r";\allowbreak{}")
    escaped = escaped.replace(", ", r",\allowbreak{} ")
    return escaped


def write_report_tex(
    output_dir: Path,
    records: Sequence[PlotRecord],
    block_design_filename: str | None,
) -> Path:
    tex_path = output_dir / "neuraedge_deep_debug_report.tex"
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{graphicx}",
        r"\usepackage{pdfpages}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\title{NeuraEdge Universal: Deep Debug Plots and Waveforms}",
        r"\author{Generated from CIFAR-10, XSim, compiler, and Vivado artifacts}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        r"\section{Evidence rule}",
        "Every page names its source artifact. CIFAR-10 plots use all 10,000 official test samples. Waveforms are generated by Vivado 2026.1 XSim from repository testbenches. Timing and power pages are included only when the corresponding routed report exists. A Vivado estimate is not relabeled as a physical board measurement.",
    ]
    if block_design_filename is not None:
        lines.extend(
            [
                r"\clearpage",
                r"\section{Native Vivado PYNQ block design}",
                "This page is exported directly from the validated system.bd block design by Vivado 2026.1. It is not a redrawn or inferred schematic.",
                rf"\includepdf[pages=-,fitpaper=true,angle=90,pagecommand={{}}]{{{block_design_filename}}}",
            ]
        )
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                r"\clearpage",
                rf"\section{{{latex_escape(record.title)}}}",
                rf"\begin{{sloppypar}}\small\textbf{{Evidence:}} {latex_breakable(record.evidence)}\end{{sloppypar}}",
                r"\begin{center}",
                rf"\includegraphics[width=0.98\textwidth,height=0.78\textheight,keepaspectratio]{{plots/{record.filename}}}",
                r"\end{center}",
            ]
        )
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex_path


def generate(
    output_dir: Path,
    timing_dir: Path | None,
    waveform_dir: Path | None = None,
    saif_power_report: Path | None = None,
    block_design: Path | None = None,
) -> tuple[Path, Path, Path]:
    root = root_dir()
    timing_dir = timing_dir or root / "outputs/pynq_full"
    waveform_dir = waveform_dir or root / "outputs/waveforms_final"
    saif_power_report = saif_power_report or (
        root / "outputs/power_activity_full/vivado_reports/routed_saif_power.rpt"
    )
    block_design = block_design or root / "outputs/pynq_full/neuraedge_pynq_block_design.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "neuraedge_deep_debug_plots.pdf"
    records: list[PlotRecord] = []
    index = 1

    reference = load_json(root / "outputs/spikingformer_reference/spikingformer_evaluation.json")
    quantized = load_json(root / "outputs/spikingformer_hw_q10_tensor_exact/spikingformer_evaluation.json")
    execution = load_json(root / "outputs/spikingformer_reference/spikingformer_execution_manifest.json")
    parameters = load_json(root / "outputs/spikingformer_hardware_parameters/spikingformer_neuraedge_weights.json")
    compiler_benchmark = load_json(root / "outputs/benchmarks/compiler_block3_memory.json")
    training_coverage_path = root / "outputs/training/spikingformer_training_coverage.json"
    training_coverage = (
        load_json(training_coverage_path) if training_coverage_path.exists() else None
    )
    forward_validation_path = root / "outputs/regression/real_full_graph_forward_validation.json"
    forward_validation = (
        load_json(forward_validation_path) if forward_validation_path.exists() else None
    )
    shard_manifest_path = root / "outputs/cifar10_shards/cifar10_shards_manifest.json"
    shard_manifest = load_json(shard_manifest_path) if shard_manifest_path.exists() else None
    float_npz = np.load(root / "outputs/spikingformer_reference/spikingformer_predictions.npz")
    quant_npz = np.load(root / "outputs/spikingformer_hw_q10_tensor_exact/spikingformer_predictions.npz")
    summary_path = timing_dir / "full_implementation_summary.txt"
    if not summary_path.exists():
        summary_path = timing_dir / "implementation_summary.txt"
    implementation = parse_implementation_summary(summary_path)
    vectorless_power_path = timing_dir / "full_vectorless_power.rpt"
    if not vectorless_power_path.exists():
        vectorless_power_path = timing_dir / "routed_vectorless_power.rpt"
    vectorless_power = parse_power_report(vectorless_power_path)
    saif_power = parse_power_report(saif_power_report)
    block_design_filename: str | None = None
    if block_design.exists():
        copied_block_design = output_dir / "pynq-block-design.pdf"
        shutil.copy2(block_design, copied_block_design)
        block_design_filename = copied_block_design.name

    with PdfPages(pdf_path) as pdf:
        summary_lines = [
            f"Dataset: {reference['dataset']} ({reference['samples']:,} samples)",
            f"Checkpoint: {reference['correct']:,}/{reference['samples']:,} = {reference['top1_accuracy_percent']:.2f}%",
            f"Exported integer arithmetic: {quantized['correct']:,}/{quantized['samples']:,} = {quantized['top1_accuracy_percent']:.2f}%",
            f"Prediction disagreements: {int(np.count_nonzero(float_npz['predictions'] != quant_npz['predictions'])):,}",
            f"Model MAC equivalents: {execution['per_image_totals']['mac_equivalents_after_exact_temporal_hoist']:,}",
            f"Tile commands: {execution['per_image_totals']['tensor_commands']:,}",
            f"Hardware parameter archive: {parameters['binary_bytes']:,} bytes, SHA-256 {parameters['binary_sha256']}",
            "Power pages use Vivado report data only; no physical-board value is inferred.",
        ]
        if implementation:
            summary_lines.extend(
                [
                    f"Full PYNQ routed setup/hold slack: {implementation.get('setup_wns_ns', float('nan')):+.3f} / {implementation.get('hold_whs_ns', float('nan')):+.3f} ns",
                    f"Clock domains: accelerator {implementation.get('accelerator_clock_mhz', float('nan')):.0f} MHz, DMA {implementation.get('dma_clock_mhz', float('nan')):.0f} MHz, control {implementation.get('control_clock_mhz', float('nan')):.0f} MHz",
                ]
            )
        if saif_power:
            summary_lines.append(
                f"Full-system SAIF estimate: {saif_power['total']:.3f} W total, {saif_power['dynamic']:.3f} W dynamic, confidence {saif_power['confidence']}, nets matched {saif_power['matched']}"
            )
        figure = text_page("Measured Evidence Summary", summary_lines)
        index = save_figure(figure, plot_dir, pdf, records, index, "evidence_summary", "Measured evidence summary",
                            "evaluation JSON, execution manifest, parameter manifest")

        if forward_validation:
            validation_lines = [
                f"Result: {forward_validation['result']}",
                f"Official CIFAR-10 fixture: {forward_validation['dataset_sample']}",
                f"Label / predicted class: {forward_validation['label']} / {forward_validation['predicted_class']}",
                f"Complete graph commands: {forward_validation['command_count']:,}",
                f"Input / output stream beats: {forward_validation['input_beats']:,} / {forward_validation['output_beats']:,}",
                f"Maximum logit delta from archived 10,000-image evaluation: {forward_validation['maximum_absolute_logit_delta_from_archived_10000_image_evaluation']:.3e}",
                f"Parameter bytes / SHA-256: {forward_validation['parameter_binary_bytes']:,} / {forward_validation['parameter_binary_sha256']}",
                f"Host trace elapsed time: {forward_validation['elapsed_seconds']:.3f} s",
            ]
            figure = text_page("Complete Exported-Graph Trace Validation", validation_lines)
            index = save_figure(
                figure, plot_dir, pdf, records, index, "full_graph_validation",
                "Complete exported-graph trace validation", str(forward_validation_path),
            )

        artifact_lines: list[str] = []
        for artifact_name in (
            "neuraedge_pynq_full.bit", "neuraedge_pynq_full.hwh",
            "neuraedge_pynq_full.xsa", "neuraedge_pynq_full_routed.dcp",
        ):
            artifact_path = timing_dir / artifact_name
            if artifact_path.exists():
                size, digest = file_evidence(artifact_path)
                artifact_lines.append(f"{artifact_name}: {size:,} bytes")
                artifact_lines.append(f"  SHA-256 {digest}")
        if artifact_lines:
            figure = text_page("Final PYNQ Deliverable Integrity", artifact_lines)
            index = save_figure(
                figure, plot_dir, pdf, records, index, "artifact_integrity",
                "Final PYNQ artifact integrity", str(timing_dir),
            )

        float_cm = np.asarray(reference["confusion_matrix_rows_true_columns_predicted"], dtype=np.int64)
        quant_cm = np.asarray(quantized["confusion_matrix_rows_true_columns_predicted"], dtype=np.int64)
        for slug, title, matrix, signed in (
            ("float_confusion", "Checkpoint confusion matrix: 10,000 images", float_cm, False),
            ("quant_confusion", "Exported integer confusion matrix: 10,000 images", quant_cm, False),
            ("confusion_delta", "Integer minus checkpoint confusion counts", quant_cm - float_cm, True),
        ):
            index = save_figure(confusion_figure(matrix, title, signed), plot_dir, pdf, records, index, slug, title,
                                "full CIFAR-10 prediction NPZ/JSON")

        figure, axis = plt.subplots(figsize=(11, 5.8))
        x = np.arange(10)
        width = 0.39
        float_acc = np.diag(float_cm) / float_cm.sum(axis=1) * 100
        quant_acc = np.diag(quant_cm) / quant_cm.sum(axis=1) * 100
        axis.bar(x - width / 2, float_acc, width, label="checkpoint", color=COLORS[0])
        axis.bar(x + width / 2, quant_acc, width, label="integer export", color=COLORS[1])
        axis.set_xticks(x, CLASSES, rotation=32, ha="right")
        axis.set_ylim(80, 100)
        axis.legend(frameon=False)
        style_axes(axis, "Per-class top-1 accuracy", ylabel="Accuracy (%)")
        index = save_figure(figure, plot_dir, pdf, records, index, "class_accuracy", "Per-class top-1 accuracy",
                            "full CIFAR-10 confusion matrices")

        labels = float_npz["labels"].astype(np.int64)
        float_logits = float_npz["logits"].astype(np.float64)
        quant_logits = quant_npz["logits"].astype(np.float64)
        float_sorted = np.sort(float_logits, axis=1)
        quant_sorted = np.sort(quant_logits, axis=1)
        float_margin = float_sorted[:, -1] - float_sorted[:, -2]
        quant_margin = quant_sorted[:, -1] - quant_sorted[:, -2]
        figure, axis = plt.subplots(figsize=(10.8, 5.8))
        axis.hist(float_margin, bins=100, density=True, alpha=0.58, label="checkpoint", color=COLORS[0])
        axis.hist(quant_margin, bins=100, density=True, alpha=0.52, label="integer export", color=COLORS[2])
        axis.set_xlim(0, np.percentile(np.concatenate([float_margin, quant_margin]), 99.5))
        axis.legend(frameon=False)
        style_axes(axis, "Top-1 versus runner-up confidence margin", "Logit margin", "Probability density")
        index = save_figure(figure, plot_dir, pdf, records, index, "margin_distribution", "Classifier margin distribution",
                            "all 10,000 checkpoint and integer logits")

        logit_error = quant_logits - float_logits
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(logit_error.ravel(), bins=150, color=COLORS[4], alpha=0.8)
        style_axes(axes[0], "All logit errors", "Integer - checkpoint logit", "Count")
        axes[1].boxplot([logit_error[:, index] for index in range(10)], tick_labels=CLASSES, showfliers=False)
        axes[1].tick_params(axis="x", rotation=40)
        style_axes(axes[1], "Error by output class", ylabel="Logit error")
        index = save_figure(figure, plot_dir, pdf, records, index, "logit_error", "Integer-export logit error",
                            "paired 10,000 x 10 logit arrays")

        disagreements = float_npz["predictions"] != quant_npz["predictions"]
        transition = np.zeros((10, 10), dtype=np.int64)
        for before, after in zip(float_npz["predictions"][disagreements], quant_npz["predictions"][disagreements]):
            transition[int(before), int(after)] += 1
        index = save_figure(confusion_figure(transition, "Prediction changes: checkpoint class to integer class"),
                            plot_dir, pdf, records, index, "prediction_changes", "Prediction disagreement transitions",
                            f"{int(disagreements.sum())} changed predictions from paired NPZ files")

        activation_stats = reference["activation_statistics"]
        spike_items = [(name, values) for name, values in activation_stats.items() if values["is_spike"]]
        spike_items.sort(key=lambda item: item[1]["nonzero_fraction"], reverse=True)
        figure, axis = plt.subplots(figsize=(12, 8))
        names = [item[0] for item in spike_items]
        rates = [item[1]["nonzero_fraction"] * 100 for item in spike_items]
        axis.barh(np.arange(len(names)), rates, color=COLORS[0])
        axis.set_yticks(np.arange(len(names)), names, fontsize=7)
        axis.invert_yaxis()
        style_axes(axis, "Measured spike density by activation", "Nonzero spike fraction (%)")
        index = save_figure(figure, plot_dir, pdf, records, index, "spike_density", "Measured spike density",
                            "streaming activation statistics over all CIFAR-10 test images")

        figure, axis = plt.subplots(figsize=(11, 6.2))
        qkv_names = [name for name in activation_stats if re.search(r"block\.\d+\.attn\.[qkv]_lif$", name)]
        x = np.arange(4)
        for offset, suffix, color in ((-0.25, "q_lif", COLORS[0]), (0, "k_lif", COLORS[1]), (0.25, "v_lif", COLORS[2])):
            values = [activation_stats[f"block.{block}.attn.{suffix}"]["nonzero_fraction"] * 100 for block in range(4)]
            axis.bar(x + offset, values, 0.24, label=suffix[0].upper(), color=color)
        axis.set_xticks(x, [f"Transformer block {block}" for block in range(4)])
        axis.legend(frameon=False)
        style_axes(axis, "Q/K/V spike density across depth", ylabel="Spike density (%)")
        index = save_figure(figure, plot_dir, pdf, records, index, "qkv_density", "Q/K/V spike density",
                            "full-test streaming activation statistics")

        nonspike_items = [(name, values) for name, values in activation_stats.items() if not values["is_spike"]]
        figure, axis = plt.subplots(figsize=(12, 7))
        names = [item[0] for item in nonspike_items]
        means = np.asarray([item[1]["mean"] for item in nonspike_items])
        stds = np.asarray([item[1]["standard_deviation"] for item in nonspike_items])
        axis.errorbar(np.arange(len(names)), means, yerr=stds, fmt="o", markersize=3, capsize=2, color=COLORS[4])
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_xticks(np.arange(len(names)), names, rotation=75, ha="right", fontsize=6.5)
        style_axes(axis, "Non-spiking activation mean and one standard deviation", ylabel="Activation value")
        index = save_figure(figure, plot_dir, pdf, records, index, "activation_moments", "Activation mean and spread",
                            "full-test streaming activation moments")

        operators = execution["operators"]
        operator_names = [operator["name"] for operator in operators if operator.get("matrix")]
        macs = np.asarray([operator["matrix"]["mac_equivalents"] for operator in operators if operator.get("matrix")])
        commands = np.asarray([operator["matrix"]["commands"] for operator in operators if operator.get("matrix")])
        input_bytes = np.asarray([operator["matrix"]["input_stream_bytes"] for operator in operators if operator.get("matrix")])
        output_bytes = np.asarray([operator["matrix"]["output_stream_bytes"] for operator in operators if operator.get("matrix")])
        for values, slug, title, ylabel, log_scale in (
            (macs, "operator_macs", "Model work by learned operator", "MAC equivalents", True),
            (commands, "operator_commands", "Tile commands by learned operator", "Commands", True),
        ):
            figure, axis = plt.subplots(figsize=(12, 7))
            order = np.argsort(values)
            axis.barh(np.arange(len(order)), values[order], color=COLORS[0 if slug.endswith("macs") else 1])
            axis.set_yticks(np.arange(len(order)), [operator_names[item] for item in order], fontsize=7)
            if log_scale:
                axis.set_xscale("log")
            style_axes(axis, title, ylabel)
            index = save_figure(figure, plot_dir, pdf, records, index, slug, title,
                                "execution manifest generated from exported tensor shapes")

        figure, axis = plt.subplots(figsize=(12, 7))
        order = np.argsort(input_bytes + output_bytes)
        axis.barh(np.arange(len(order)), input_bytes[order] / 1e6, label="input", color=COLORS[0])
        axis.barh(np.arange(len(order)), output_bytes[order] / 1e6, left=input_bytes[order] / 1e6,
                  label="output", color=COLORS[3])
        axis.set_yticks(np.arange(len(order)), [operator_names[item] for item in order], fontsize=7)
        axis.legend(frameon=False)
        style_axes(axis, "Raw tile-stream traffic by learned operator", "Traffic (MB per image)")
        index = save_figure(figure, plot_dir, pdf, records, index, "operator_traffic", "Operator stream traffic",
                            "execution manifest; excludes inter-layer DDR spill")

        arithmetic = arithmetic_groups(operators)
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.pie(list(arithmetic.values()), labels=list(arithmetic), autopct="%1.1f%%", colors=COLORS[: len(arithmetic)],
                 startangle=90, wedgeprops={"linewidth": 0.7, "edgecolor": "white"})
        axis.set_title("MAC-equivalent share by arithmetic mode", fontsize=13, fontweight="bold")
        index = save_figure(figure, plot_dir, pdf, records, index, "arithmetic_share", "Arithmetic mode share",
                            "execution manifest model-equivalent operation counts")

        matrix_records = [(operator["name"], operator["matrix"]) for operator in operators if operator.get("matrix")]
        figure, axis = plt.subplots(figsize=(9.5, 6.5))
        for arithmetic_name in sorted({item[1]["arithmetic"] for item in matrix_records}):
            subset = [item for item in matrix_records if item[1]["arithmetic"] == arithmetic_name]
            axis.scatter([item[1]["reduction"] for item in subset], [item[1]["columns"] for item in subset],
                         s=[max(25, math.sqrt(item[1]["rows"])) for item in subset], label=arithmetic_name, alpha=0.75)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.legend(frameon=False)
        style_axes(axis, "Matrix shape landscape", "Reduction K", "Output columns N")
        index = save_figure(figure, plot_dir, pdf, records, index, "matrix_shapes", "Matrix shape landscape",
                            "execution manifest; marker area tracks output-row count")

        projection_names = [projection["name"] for projection in parameters["projections"]]
        weight_scales = [projection["weight_scale"] for projection in parameters["projections"]]
        factor_errors = [projection["represented_factor_maximum_error"] for projection in parameters["projections"]]
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(weight_scales, marker="o", markersize=3, color=COLORS[0])
        axes[0].set_yscale("log")
        style_axes(axes[0], "Per-projection INT8 weight scale", ylabel="Scale")
        axes[1].plot(factor_errors, marker="o", markersize=3, color=COLORS[2])
        axes[1].set_yscale("log")
        axes[1].set_xticks(range(len(projection_names)), projection_names, rotation=75, ha="right", fontsize=6.5)
        style_axes(axes[1], "Maximum represented requantization-factor error", ylabel="Absolute factor error")
        index = save_figure(figure, plot_dir, pdf, records, index, "projection_quantization", "Projection quantization diagnostics",
                            "hashed hardware parameter manifest")

        binary_path = root / "outputs/spikingformer_hardware_parameters" / parameters["binary_file"]
        binary = binary_path.read_bytes()
        all_weights: list[np.ndarray] = []
        for projection in parameters["projections"]:
            section = projection["sections"][0]
            if "weight" in section["name"] and section["dtype"] == "|i1":
                all_weights.append(np.frombuffer(binary, dtype=np.int8, count=section["length"], offset=section["offset"]))
        head_section = parameters["head"]["sections"][0]
        all_weights.append(np.frombuffer(binary, dtype=np.int8, count=head_section["length"], offset=head_section["offset"]))
        weight_values = np.concatenate(all_weights)
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        axes[0].hist(weight_values, bins=np.arange(-128.5, 128.5), color=COLORS[0])
        axes[0].set_yscale("log")
        style_axes(axes[0], "All exported INT8 weights", "Integer weight", "Count (log)")
        unique, counts = np.unique(weight_values, return_counts=True)
        top = np.argsort(counts)[-20:]
        axes[1].bar(unique[top].astype(str), counts[top], color=COLORS[1])
        axes[1].tick_params(axis="x", rotation=60)
        style_axes(axes[1], "Twenty most frequent codes", "Integer weight", "Count")
        index = save_figure(figure, plot_dir, pdf, records, index, "weight_distribution", "Exported weight distribution",
                            "direct section views from the SHA-256-checked parameter binary")

        figure, axes = plt.subplots(1, 2, figsize=(11, 5))
        names = ["scalar packer", "vectorized packer"]
        seconds = [95.601, compiler_benchmark["compile_seconds"]]
        peaks = [71.468169, compiler_benchmark["tracemalloc_peak_bytes"] / 1e6]
        axes[0].bar(names, seconds, color=[COLORS[2], COLORS[1]])
        axes[0].set_yscale("log")
        style_axes(axes[0], "Block-3 compile time", ylabel="Seconds (log)")
        axes[1].bar(names, peaks, color=[COLORS[2], COLORS[1]])
        style_axes(axes[1], "Peak traced memory", ylabel="MB")
        index = save_figure(figure, plot_dir, pdf, records, index, "compiler_benchmark", "Largest-layer compiler benchmark",
                            "recorded scalar baseline and outputs/benchmarks/compiler_block3_memory.json")

        if training_coverage:
            training_summary = training_coverage["summary"]
            semantics = training_coverage["training_semantics"]
            training_lines = [
                f"Target: {semantics['target_graph']}",
                f"Execution operators covered: {training_summary['covered_execution_operator_count']}/{training_summary['execution_operator_count']}; uncovered {training_summary['uncovered_operator_count']}",
                f"Learned affine groups / dynamic attention products: {training_summary['learned_affine_group_count']} / {training_summary['dynamic_attention_matmul_count']}",
                f"Trainable deployed parameter elements: {training_summary['trainable_deployed_parameter_elements']:,}",
                f"Backward partial tiles / signed multiplications: {training_summary['total_backward_partial_tiles']:,} / {(training_summary['learned_backward_multiplications'] + training_summary['dynamic_backward_multiplications']):,}",
                f"Fabric SGDW update commands: {training_summary['sgdw_update_commands']:,}",
                f"Quantizer derivative: {semantics['quantizer_gradient']}",
                f"LIF derivative: {semantics['lif_gradient']}",
                f"Temporal schedule: {semantics['temporal_method']}",
                f"Optimizer: {semantics['optimizer']}",
                f"Parameter residency: {semantics['parameter_residency']}",
            ]
            figure = text_page("Full-Backbone Training Schedule", training_lines)
            index = save_figure(
                figure, plot_dir, pdf, records, index, "training_schedule",
                "Full-backbone training schedule and numeric contract", str(training_coverage_path),
            )

            learned = training_coverage["learned_affine_coverage"]
            learned_names = [item["name"] for item in learned]
            weight_elements = np.asarray([item["weight_elements"] for item in learned], dtype=np.int64)
            bias_elements = np.asarray([item["bias_elements"] for item in learned], dtype=np.int64)
            order = np.argsort(weight_elements + bias_elements)
            figure, axis = plt.subplots(figsize=(12, 8.5))
            axis.barh(np.arange(len(order)), weight_elements[order], label="weight", color=COLORS[0])
            axis.barh(
                np.arange(len(order)), bias_elements[order], left=weight_elements[order],
                label="bias", color=COLORS[3],
            )
            axis.set_yticks(np.arange(len(order)), [learned_names[item] for item in order], fontsize=7)
            axis.set_xscale("log")
            axis.legend(frameon=False)
            style_axes(axis, "Trainable deployed parameters by affine group", "Parameter elements (log)")
            index = save_figure(
                figure, plot_dir, pdf, records, index, "training_parameters",
                "Trainable deployed parameter distribution", str(training_coverage_path),
            )

            backward_tiles = np.asarray(
                [item["backward_partial_tiles"] for item in learned], dtype=np.int64
            )
            backward_multiplies = np.asarray(
                [item["backward_multiplications"] for item in learned], dtype=np.int64
            )
            order = np.argsort(backward_multiplies)
            figure, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=True)
            y = np.arange(len(order))
            axes[0].barh(y, backward_tiles[order], color=COLORS[1])
            axes[0].set_yticks(y, [learned_names[item] for item in order], fontsize=7)
            axes[0].set_xscale("log")
            style_axes(axes[0], "Backward partial tiles", "Tiles per sample (log)")
            axes[1].barh(y, backward_multiplies[order], color=COLORS[2])
            axes[1].set_xscale("log")
            style_axes(axes[1], "Signed backward multiplications", "Multiplications per sample (log)")
            index = save_figure(
                figure, plot_dir, pdf, records, index, "training_backward_work",
                "Exact learned-layer backward work", str(training_coverage_path),
            )

            backward_sites = {
                "affine dX/dW/dB": training_summary["learned_affine_group_count"],
                "dynamic QK/AV": training_summary["dynamic_attention_matmul_count"],
                "LIF surrogate": training_summary["lif_node_count"],
                "max-pool scatter": training_summary["maxpool_backward_count"],
                "residual fan-out": training_summary["residual_gradient_fanout_count"],
                "global pool": 1,
            }
            figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
            axes[0].bar(
                np.arange(len(backward_sites)), list(backward_sites.values()),
                color=COLORS[:len(backward_sites)],
            )
            axes[0].set_xticks(
                np.arange(len(backward_sites)), list(backward_sites), rotation=35, ha="right"
            )
            style_axes(axes[0], "Implemented backward sites", ylabel="Site count")
            work_labels = ["learned affine", "dynamic attention"]
            work_values = [
                training_summary["learned_backward_multiplications"],
                training_summary["dynamic_backward_multiplications"],
            ]
            axes[1].pie(
                work_values, labels=work_labels, autopct="%1.2f%%", colors=[COLORS[0], COLORS[3]],
                startangle=90, wedgeprops={"linewidth": 0.7, "edgecolor": "white"},
            )
            axes[1].set_title("Backward multiplication share", fontsize=13, fontweight="bold")
            index = save_figure(
                figure, plot_dir, pdf, records, index, "training_operator_coverage",
                "Backward operator coverage and work composition", str(training_coverage_path),
            )

        if shard_manifest:
            split_names: list[str] = []
            shard_indices: list[int] = []
            shard_sizes: list[float] = []
            for split_name, shards in shard_manifest["splits"].items():
                for shard_index, shard in enumerate(shards):
                    split_names.append(split_name)
                    shard_indices.append(shard_index)
                    shard_sizes.append(shard["bytes"] / 1e6)
            figure, axis = plt.subplots(figsize=(11.5, 5.8))
            for color_index, split_name in enumerate(sorted(set(split_names))):
                selected = [index for index, name in enumerate(split_names) if name == split_name]
                axis.plot(
                    [shard_indices[index] for index in selected],
                    [shard_sizes[index] for index in selected],
                    marker="o", markersize=3, label=split_name, color=COLORS[color_index],
                )
            axis.legend(frameon=False)
            style_axes(axis, "Official CIFAR-10 shard sizes", "Shard index", "Compressed NPZ size (MB)")
            index = save_figure(
                figure, plot_dir, pdf, records, index, "training_dataset_shards",
                "Deterministic CIFAR-10 board-training shards", str(shard_manifest_path),
            )

        waveform_manifest_path = waveform_dir / "waveform_manifest.json"
        if waveform_manifest_path.exists():
            waveform_manifest = load_json(waveform_manifest_path)
            for capture in waveform_manifest["captures"]:
                vcd_path = waveform_dir / capture["vcd"]
                figure, selected_names = waveform_figure(vcd_path, f"XSim: {capture['testbench']}")
                evidence = (
                    f"Vivado 2026.1 XSim VCD, requested window {capture['duration']} {capture['unit']}; "
                    f"signals: {', '.join(name.rsplit('/', 1)[-1] for name in selected_names)}"
                )
                index = save_figure(figure, plot_dir, pdf, records, index,
                                    f"wave_{capture['testbench'].removeprefix('tb_neuraedge_')}",
                                    f"Waveform: {capture['testbench']}", evidence)

        if timing_dir is not None and timing_dir.exists():
            timing_summary_path = timing_dir / "full_routed_timing_summary.rpt"
            if not timing_summary_path.exists():
                timing_summary_path = timing_dir / "routed_timing_summary.rpt"
            clock_timing = parse_clock_timing(timing_summary_path)
            if clock_timing:
                names = sorted(clock_timing)
                setup = [clock_timing[name][0] for name in names]
                hold = [clock_timing[name][1] for name in names]
                figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
                axes[0].bar(names, setup, color=COLORS[: len(names)])
                axes[0].axhline(0, color=COLORS[2], linewidth=1.0)
                style_axes(axes[0], "Setup slack by clock domain", ylabel="WNS (ns)")
                axes[1].bar(names, hold, color=COLORS[: len(names)])
                axes[1].axhline(0, color=COLORS[2], linewidth=1.0)
                style_axes(axes[1], "Hold slack by clock domain", ylabel="WHS (ns)")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "clock_domain_timing",
                    "Routed timing by clock domain", str(timing_summary_path),
                )

            utilization_path = timing_dir / "full_routed_utilization_hierarchical.rpt"
            if not utilization_path.exists():
                utilization_path = timing_dir / "routed_utilization_hierarchical.rpt"
            utilization = parse_utilization_hierarchy(utilization_path)
            if "system_wrapper" in utilization:
                top = utilization["system_wrapper"]
                labels_resource = ["LUT", "FF", "DSP48", "BRAM tile"]
                counts_resource = [top["lut"], top["ff"], top["dsp"], top["bram36"] + top["bram18"] / 2]
                capacities = [53_200, 106_400, 220, 140]
                percents = [100 * count / capacity for count, capacity in zip(counts_resource, capacities)]
                figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
                axes[0].bar(labels_resource, counts_resource, color=COLORS[:4])
                axes[0].set_yscale("log")
                style_axes(axes[0], "Implemented primitive use", ylabel="Count (log scale)")
                axes[1].bar(labels_resource, percents, color=COLORS[:4])
                axes[1].set_ylim(0, 100)
                for position, value in enumerate(percents):
                    axes[1].text(position, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
                style_axes(axes[1], "PYNQ-Z2 device utilization", ylabel="Available resources used (%)")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "full_routed_resources",
                    "Full PYNQ routed resource utilization", str(utilization_path),
                )

                hierarchy_order = [
                    name for name in (
                        "accelerator", "dma", "control_interconnect", "memory_interconnect",
                        "accelerator_control_cdc", "dma_control_cdc", "mm2s_cdc", "s2mm_cdc",
                    ) if name in utilization
                ]
                figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
                for axis, key, title in zip(
                    axes.ravel(), ("lut", "ff", "dsp", "bram36"),
                    ("LUTs", "Flip-flops", "DSP48 blocks", "RAMB36 blocks"),
                ):
                    values = [utilization[name][key] for name in hierarchy_order]
                    axis.barh(np.arange(len(hierarchy_order)), values, color=COLORS[0])
                    axis.set_yticks(np.arange(len(hierarchy_order)), hierarchy_order, fontsize=8)
                    axis.invert_yaxis()
                    style_axes(axis, title, xlabel="Primitive count")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "resource_hierarchy",
                    "Resource use by full-system hierarchy", str(utilization_path),
                )

            drc_path = timing_dir / "full_drc.rpt"
            drc_rows = parse_drc_summary(drc_path)
            if drc_rows:
                figure, axis = plt.subplots(figsize=(11, 6.2))
                labels_drc = [f"{rule}: {description}" for rule, _, description, _ in drc_rows]
                counts_drc = [count for _, _, _, count in drc_rows]
                axis.barh(np.arange(len(drc_rows)), counts_drc, color=COLORS[3])
                axis.set_yticks(np.arange(len(drc_rows)), labels_drc, fontsize=8)
                axis.invert_yaxis()
                axis.set_xscale("log")
                style_axes(axis, "Vivado DRC advisory classification", "Reported checks (log scale)")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "drc_classification",
                    "DRC warning classification", str(drc_path),
                )

            cdc_path = timing_dir / "full_cdc.rpt"
            cdc_rows = parse_cdc_summary(cdc_path)
            if cdc_rows:
                figure, axis = plt.subplots(figsize=(10.5, 5.4))
                labels_cdc = [f"{rule} {severity}\n{description}" for rule, severity, _, description in cdc_rows]
                counts_cdc = [count for _, _, count, _ in cdc_rows]
                axis.bar(labels_cdc, counts_cdc, color=[COLORS[1], COLORS[3]][: len(cdc_rows)])
                style_axes(axis, "Vivado clock-domain crossing classification", ylabel="Reported crossings")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "cdc_classification",
                    "Clock-domain crossing classification", str(cdc_path),
                )

            if vectorless_power and saif_power:
                labels_power = ["Total", "Dynamic", "Static"]
                vector_values = [vectorless_power[key] for key in ("total", "dynamic", "static")]
                saif_values = [saif_power[key] for key in ("total", "dynamic", "static")]
                x = np.arange(len(labels_power))
                figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
                axes[0].bar(x - 0.2, vector_values, 0.4, label="vectorless", color=COLORS[3])
                axes[0].bar(x + 0.2, saif_values, 0.4, label="SAIF annotated", color=COLORS[1])
                axes[0].set_xticks(x, labels_power)
                axes[0].legend(frameon=False)
                style_axes(axes[0], "Full-system Vivado power estimates", ylabel="Power (W)")
                hierarchy_order = [
                    name for name in ("accelerator", "dma", "mm2s_cdc", "s2mm_cdc")
                    if name in vectorless_power["hierarchy"] and name in saif_power["hierarchy"]
                ]
                hierarchy_x = np.arange(len(hierarchy_order))
                axes[1].bar(
                    hierarchy_x - 0.2,
                    [vectorless_power["hierarchy"][name] for name in hierarchy_order],
                    0.4, label="vectorless", color=COLORS[3],
                )
                axes[1].bar(
                    hierarchy_x + 0.2,
                    [saif_power["hierarchy"][name] for name in hierarchy_order],
                    0.4, label="SAIF annotated", color=COLORS[1],
                )
                axes[1].set_xticks(hierarchy_x, hierarchy_order, rotation=25, ha="right")
                axes[1].legend(frameon=False)
                style_axes(axes[1], "Dynamic power by hierarchy", ylabel="Dynamic power (W)")
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "power_comparison",
                    "Vectorless and SAIF-annotated power estimates",
                    f"{vectorless_power_path}; {saif_power_report}; SAIF match {saif_power['matched']}",
                )

                power_lines = [
                    f"Vectorless: total {vectorless_power['total']:.3f} W, dynamic {vectorless_power['dynamic']:.3f} W, confidence {vectorless_power['confidence']}",
                    f"SAIF: total {saif_power['total']:.3f} W, dynamic {saif_power['dynamic']:.3f} W, confidence {saif_power['confidence']}",
                    f"SAIF implementation nets matched: {saif_power['matched']}",
                    "Captured tests do not initialize every memory location; Vivado reports X-state warnings for untouched storage.",
                    "These are tool estimates for the routed design, not measurements from a PYNQ-Z2 power rail.",
                ]
                figure = text_page("Power Evidence and Confidence Boundary", power_lines)
                index = save_figure(
                    figure, plot_dir, pdf, records, index, "power_confidence",
                    "Power confidence and annotation coverage", str(saif_power_report),
                )

            timing_path = timing_dir / "full_routed_timing_paths_250.rpt"
            if not timing_path.exists():
                timing_path = timing_dir / "routed_timing_paths_250.rpt"
            if timing_path.exists():
                slacks = parse_timing_slacks(timing_path)
                if slacks.size:
                    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
                    axes[0].hist(slacks, bins=min(40, max(8, slacks.size // 5)), color=COLORS[0])
                    axes[0].axvline(0, color=COLORS[2], linewidth=1.2)
                    style_axes(axes[0], "Reported routed path slack", "Setup slack (ns)", "Path count")
                    axes[1].plot(np.sort(slacks), color=COLORS[4])
                    axes[1].axhline(0, color=COLORS[2], linewidth=1.2)
                    style_axes(axes[1], "Sorted critical-path slack", "Path rank", "Setup slack (ns)")
                    index = save_figure(figure, plot_dir, pdf, records, index, "routed_timing", "Routed timing path distribution",
                                        str(timing_path))

    tex_path = write_report_tex(output_dir, records, block_design_filename)
    manifest_path = output_dir / "neuraedge_debug_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "neuraedge-deep-debug-v1",
                "plot_count": len(records),
                "plots": [record.__dict__ for record in records],
                "multipage_pdf": pdf_path.name,
                "latex": tex_path.name,
                "native_vivado_block_design": block_design_filename,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return pdf_path, tex_path, manifest_path


def compile_tex(tex_path: Path, pdflatex: str) -> Path:
    for _ in range(2):
        subprocess.run([pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                       cwd=tex_path.parent, check=True)
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root_dir() / "outputs/deep_debug")
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--waveform-dir", type=Path)
    parser.add_argument("--saif-power-report", type=Path)
    parser.add_argument("--block-design", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pdflatex", default="pdflatex")
    args = parser.parse_args()
    pdf_path, tex_path, manifest_path = generate(
        args.output_dir.resolve(),
        args.timing_dir.resolve() if args.timing_dir else None,
        args.waveform_dir.resolve() if args.waveform_dir else None,
        args.saif_power_report.resolve() if args.saif_power_report else None,
        args.block_design.resolve() if args.block_design else None,
    )
    print(f"plots_pdf={pdf_path}")
    print(f"latex={tex_path}")
    print(f"manifest={manifest_path}")
    if args.compile:
        print(f"report_pdf={compile_tex(tex_path, args.pdflatex)}")


if __name__ == "__main__":
    main()
