from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_power_report(path: Path) -> dict[str, float | str]:
    text = path.read_text(errors="ignore")
    patterns = {
        "total_on_chip_w": r"Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)",
        "dynamic_w": r"Dynamic \(W\)\s*\|\s*([0-9.]+)",
        "device_static_w": r"Device Static \(W\)\s*\|\s*([0-9.]+)",
        "junction_temp_c": r"Junction Temperature \(C\)\s*\|\s*([0-9.]+)",
    }
    parsed: dict[str, float | str] = {"report": str(path)}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            parsed[key] = float(match.group(1))
    confidence = re.search(r"Overall confidence level\s*\|\s*([^|]+)\|", text)
    if confidence:
        parsed["confidence"] = confidence.group(1).strip()
    return parsed


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    project = repo_root() / "cifar10_neuromorphic"
    out_dir = project / "outputs_spikingformer_head_accelerator"
    activity_dir = out_dir / "activity_power"
    activity_dir.mkdir(parents=True, exist_ok=True)

    vectorless = parse_power_report(out_dir / "vivado" / "spikingformer_head_power.rpt")
    inference = parse_power_report(activity_dir / "spikingformer_head_power_inference_saif.rpt")
    learning = parse_power_report(activity_dir / "spikingformer_head_power_learning_saif.rpt")
    rtl_val = load_json(out_dir / "spikingformer_head_rtl_validation.json")
    learn_val = load_json(out_dir / "spikingformer_head_onchip_learning_validation.json")

    summary: dict[str, object] = {
        "note": (
            "Vivado-estimated FPGA power for the routed Spikingformer-head accelerator. "
            "Vectorless is Vivado's default activity estimate; inference_saif and "
            "learning_saif are annotated from XSim SAIF traces of the RTL testbenches."
        ),
        "target": {
            "part": "xc7z020clg400-1",
            "clock_ns": 10.0,
            "top": "cifar10_spikingformer_head_pynq_axi_stream_top",
            "annotated_instance": "u_head",
        },
        "validation": {
            "samples": rtl_val.get("samples"),
            "accuracy": rtl_val.get("accuracy"),
            "score_mismatch_count": rtl_val.get("score_mismatch_count"),
            "prediction_mismatch_count": rtl_val.get("prediction_mismatch_count"),
            "onchip_learning_updates": learn_val.get("updates"),
            "onchip_learning_score_mismatch_count": learn_val.get("score_mismatch_count"),
            "onchip_learning_prediction_mismatch_count": learn_val.get("prediction_mismatch_count"),
        },
        "vectorless": vectorless,
        "inference_saif": inference,
        "learning_saif": learning,
    }

    for name, data in [("inference_saif", inference), ("learning_saif", learning)]:
        dyn = data.get("dynamic_w")
        base_dyn = vectorless.get("dynamic_w")
        total = data.get("total_on_chip_w")
        base_total = vectorless.get("total_on_chip_w")
        if isinstance(dyn, float) and isinstance(base_dyn, float) and base_dyn > 0:
            summary[f"{name}_dynamic_vs_vectorless"] = dyn / base_dyn
        if isinstance(total, float) and isinstance(base_total, float) and base_total > 0:
            summary[f"{name}_total_vs_vectorless"] = total / base_total

    summary_path = activity_dir / "spikingformer_head_activity_power_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cases = [
        ("vectorless", vectorless),
        ("inference_saif", inference),
        ("learning_saif", learning),
    ]
    labels = []
    dynamic = []
    static = []
    for name, data in cases:
        if isinstance(data.get("dynamic_w"), float):
            labels.append(name)
            dynamic.append(float(data.get("dynamic_w", 0.0)))
            static.append(float(data.get("device_static_w", 0.0)))

    if labels:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(labels, dynamic, label="dynamic")
        ax.bar(labels, static, bottom=dynamic, label="device static")
        ax.set_ylabel("Vivado power (W)")
        ax.set_title("Spikingformer Head Power: Vectorless vs XSim SAIF")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        plot_path = activity_dir / "spikingformer_head_activity_power.png"
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

        pdf_path = activity_dir / "spikingformer_head_activity_power_report.pdf"
        with PdfPages(pdf_path) as pdf:
            img = plt.imread(plot_path)
            fig = plt.figure(figsize=(10, 7))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig = plt.figure(figsize=(10, 7))
            fig.text(0.06, 0.93, "Spikingformer Head Activity Power Summary", fontsize=16, weight="bold")
            fig.text(0.06, 0.85, json.dumps(summary, indent=2), fontsize=7.7, family="monospace", va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
