"""Evaluate the trained Spikingformer and capture hardware calibration data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from spikingformer_reference import load_checkpoint
from spikingformer_quantization import (
    prepare_hardware_emulation,
    quantize_cifar_input_u8,
)


CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class RunningTensorStats:
    """Numerically stable aggregate statistics retained on the source device."""

    elements: int
    batches: int
    minimum: torch.Tensor
    maximum: torch.Tensor
    total: torch.Tensor
    total_squares: torch.Tensor
    nonzero: torch.Tensor
    is_spike: bool

    @classmethod
    def from_tensor(cls, value: torch.Tensor, is_spike: bool) -> "RunningTensorStats":
        detached = value.detach().float()
        return cls(
            elements=detached.numel(),
            batches=1,
            minimum=detached.amin(),
            maximum=detached.amax(),
            total=detached.sum(dtype=torch.float64),
            total_squares=detached.square().sum(dtype=torch.float64),
            nonzero=torch.count_nonzero(detached),
            is_spike=is_spike,
        )

    def update(self, value: torch.Tensor, is_spike: bool) -> None:
        if self.is_spike != is_spike:
            raise ValueError("a tensor changed between analog and spike semantics")
        detached = value.detach().float()
        self.elements += detached.numel()
        self.batches += 1
        self.minimum = torch.minimum(self.minimum, detached.amin())
        self.maximum = torch.maximum(self.maximum, detached.amax())
        self.total += detached.sum(dtype=torch.float64)
        self.total_squares += detached.square().sum(dtype=torch.float64)
        self.nonzero += torch.count_nonzero(detached)

    def as_dict(self) -> dict[str, int | float | bool]:
        mean = float((self.total / self.elements).cpu())
        second_moment = float((self.total_squares / self.elements).cpu())
        variance = max(0.0, second_moment - mean * mean)
        return {
            "elements": self.elements,
            "batches": self.batches,
            "minimum": float(self.minimum.cpu()),
            "maximum": float(self.maximum.cpu()),
            "mean": mean,
            "standard_deviation": variance ** 0.5,
            "nonzero_fraction": float((self.nonzero / self.elements).cpu()),
            "zero_fraction": 1.0 - float((self.nonzero / self.elements).cpu()),
            "is_spike": self.is_spike,
        }


class CalibrationObserver:
    """Collect aggregate ranges and sparsity without retaining feature maps."""

    def __init__(self) -> None:
        self.stats: dict[str, RunningTensorStats] = {}

    def __call__(self, name: str, value: torch.Tensor, is_spike: bool) -> None:
        if name in self.stats:
            self.stats[name].update(value, is_spike)
        else:
            self.stats[name] = RunningTensorStats.from_tensor(value, is_spike)

    def as_dict(self) -> dict[str, dict[str, int | float | bool]]:
        return {name: item.as_dict() for name, item in self.stats.items()}


def evaluate(arguments: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(arguments.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)
    loaded = load_checkpoint(arguments.checkpoint, device=device)
    model = loaded.model
    quantization_report: dict[str, object] | None = None
    if arguments.hardware_emulation:
        model, hardware_report = prepare_hardware_emulation(
            model,
            fractional_bits=arguments.fractional_bits,
            weight_granularity=arguments.weight_granularity,
        )
        model = model.to(device)
        quantization_report = hardware_report.as_dict()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(
        root=str(Path(arguments.data_root).resolve()),
        train=False,
        transform=transform,
        download=arguments.download,
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=arguments.workers > 0,
        generator=generator,
    )
    observer = CalibrationObserver() if arguments.collect_stats else None
    confusion = np.zeros((10, 10), dtype=np.int64)
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    correct = 0
    samples = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (image, label) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if arguments.hardware_emulation:
                image, _, _ = quantize_cifar_input_u8(image)
            logits = model(image, observer=observer)
            prediction = logits.argmax(dim=1)
            labels_cpu = label.cpu().numpy()
            predictions_cpu = prediction.cpu().numpy()
            np.add.at(confusion, (labels_cpu, predictions_cpu), 1)
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(labels_cpu)
            correct += int((prediction == label).sum().item())
            samples += label.numel()
            if arguments.log_interval and (batch_index + 1) % arguments.log_interval == 0:
                print(
                    f"batch={batch_index + 1}/{len(loader)} "
                    f"samples={samples} accuracy={100.0 * correct / samples:.4f}%",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    logits_array = np.concatenate(all_logits, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    predictions_array = logits_array.argmax(axis=1)
    class_totals = confusion.sum(axis=1)
    class_correct = np.diag(confusion)
    class_accuracy = np.divide(
        class_correct,
        class_totals,
        out=np.zeros(10, dtype=np.float64),
        where=class_totals != 0,
    )
    accuracy = 100.0 * correct / samples

    np.savez_compressed(
        output_dir / "spikingformer_predictions.npz",
        logits=logits_array,
        labels=labels_array,
        predictions=predictions_array,
        confusion_matrix=confusion,
    )
    with (output_dir / "spikingformer_predictions.csv").open(
        "w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample", "label", "prediction", "correct", *(
            f"logit_{name}" for name in CIFAR10_CLASSES)))
        for index, (label, prediction, logits) in enumerate(zip(
            labels_array, predictions_array, logits_array)):
            writer.writerow((
                index,
                int(label),
                int(prediction),
                int(label == prediction),
                *(f"{float(item):.9g}" for item in logits),
            ))

    report: dict[str, object] = {
        "checkpoint": str(loaded.checkpoint_path),
        "architecture": loaded.architecture,
        "checkpoint_epoch": loaded.epoch,
        "checkpoint_recorded_metric_percent": loaded.metric,
        "dataset": "CIFAR-10 official test split",
        "samples": samples,
        "correct": correct,
        "top1_accuracy_percent": accuracy,
        "batch_size": arguments.batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "images_per_second": samples / elapsed,
        "normalization_mean": CIFAR10_MEAN,
        "normalization_std": CIFAR10_STD,
        "class_accuracy_percent": {
            name: 100.0 * float(class_accuracy[index])
            for index, name in enumerate(CIFAR10_CLASSES)
        },
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "activation_statistics": observer.as_dict() if observer else {},
        "hardware_emulation": quantization_report,
    }
    (output_dir / "spikingformer_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({
        "samples": samples,
        "correct": correct,
        "top1_accuracy_percent": accuracy,
        "elapsed_seconds": elapsed,
        "images_per_second": samples / elapsed,
        "output_dir": str(output_dir),
    }, indent=2), flush=True)
    return report


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--collect-stats", action="store_true")
    parser.add_argument("--hardware-emulation", action="store_true")
    parser.add_argument("--fractional-bits", type=int, default=10)
    parser.add_argument(
        "--weight-granularity", choices=("tensor", "channel"), default="tensor")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_arguments())
