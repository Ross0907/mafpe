"""Run resumable full-backbone Spikingformer fine-tuning on a PYNQ overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from neuraedge_pynq import NeuraEdgeDriver
from run_spikingformer_pynq import prepare_input
from spikingformer_accelerator import SpikingformerAcceleratorModel
from spikingformer_parameters import SpikingformerParameterArchive
from spikingformer_training_runtime import (
    SpikingformerOnDeviceTrainer,
    TrainableSpikingformerState,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--minimum-learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gradient-clip-q15", type=int, default=4 << 15)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--layer-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Cifar10ShardSet:
    """Validated deterministic traversal of one manifest split."""

    def __init__(self, manifest_path: Path, split: str) -> None:
        manifest_path = manifest_path.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        if manifest.get("format") != "neuraedge-official-cifar10-shards-v1":
            raise ValueError("unsupported CIFAR-10 shard manifest")
        entries = manifest.get("splits", {}).get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"CIFAR-10 manifest has no {split} shards")
        self.entries = tuple(entries)
        self.root = manifest_path.parent
        self.split = split
        self.sample_count = sum(int(entry["samples"]) for entry in self.entries)
        for entry in self.entries:
            path = self.root / str(entry["file"])
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"CIFAR-10 shard integrity failure: {path}")

    def batches(
        self,
        *,
        epoch: int,
        batch_size: int,
        seed: int,
        shuffle: bool,
        start_batch: int = 0,
    ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed + epoch * 1_000_003 + (0 if self.split == "train" else 97))
        shard_order = np.arange(len(self.entries))
        if shuffle:
            rng.shuffle(shard_order)
        global_batch = 0
        for shard_index in shard_order:
            entry = self.entries[int(shard_index)]
            with np.load(self.root / str(entry["file"]), allow_pickle=False) as shard:
                images = np.asarray(shard["images"], dtype=np.uint8)
                labels = np.asarray(shard["labels"], dtype=np.int64)
            order = np.arange(len(labels))
            if shuffle:
                rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                selected = order[start:start + batch_size]
                if global_batch >= start_batch:
                    yield global_batch, images[selected], labels[selected]
                global_batch += 1


def augment_cifar10(images: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic reflect-pad crop and horizontal flip for one batch."""

    rng = np.random.default_rng(seed)
    output = np.empty_like(images)
    for index, image in enumerate(images):
        padded = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        y = int(rng.integers(0, 9))
        x = int(rng.integers(0, 9))
        crop = padded[y:y + 32, x:x + 32]
        output[index] = crop[:, ::-1] if rng.random() < 0.5 else crop
    return output


def cosine_learning_rate(
    epoch: int,
    epochs: int,
    maximum: float,
    minimum: float,
) -> float:
    if epochs == 1:
        return minimum
    phase = epoch / float(epochs - 1)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * phase))


def evaluate(
    model: SpikingformerAcceleratorModel,
    dataset: Cifar10ShardSet,
    *,
    batch_size: int,
    seed: int,
) -> tuple[float, int, int]:
    correct = 0
    total = 0
    for _, raw_images, labels in dataset.batches(
        epoch=0, batch_size=batch_size, seed=seed, shuffle=False):
        images = prepare_input(raw_images, "raw-u8")
        prediction = model.infer(images).predicted_class
        correct += int(np.count_nonzero(prediction == labels))
        total += len(labels)
    return 100.0 * correct / total, correct, total


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if arguments.checkpoint_interval <= 0:
        raise ValueError("checkpoint-interval must be positive")
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pause_path = output_dir / "PAUSE"
    status_path = output_dir / "training_status.json"
    latest_path = output_dir / "latest_checkpoint.npz"
    best_path = output_dir / "best_checkpoint.npz"
    history_path = output_dir / "training_history.jsonl"

    archive = SpikingformerParameterArchive(arguments.manifest)
    state = TrainableSpikingformerState(archive)
    start_epoch = 0
    start_batch = 0
    best_accuracy = 0.0
    if arguments.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {latest_path}")
        metadata = state.load_checkpoint(latest_path)
        start_epoch = int(metadata["epoch"])
        start_batch = int(metadata["batch_index"])
        best_accuracy = float(metadata["best_accuracy_percent"])

    train_data = Cifar10ShardSet(arguments.data_manifest, "train")
    test_data = Cifar10ShardSet(arguments.data_manifest, "test")
    driver = NeuraEdgeDriver.from_bitfile(
        arguments.bitfile, download=not arguments.no_download)
    trainer = SpikingformerOnDeviceTrainer(
        state, driver, layer_timeout_seconds=arguments.layer_timeout_seconds)
    inference_model = SpikingformerAcceleratorModel(
        state, driver, layer_timeout_seconds=arguments.layer_timeout_seconds)

    job_started = time.time()
    for epoch in range(start_epoch, arguments.epochs):
        epoch_batch_start = start_batch if epoch == start_epoch else 0
        learning_rate = cosine_learning_rate(
            epoch,
            arguments.epochs,
            arguments.learning_rate,
            arguments.minimum_learning_rate,
        )
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        last_batch = epoch_batch_start - 1
        for batch_index, raw_images, labels in train_data.batches(
            epoch=epoch,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
            shuffle=True,
            start_batch=epoch_batch_start,
        ):
            if pause_path.exists():
                state.save_checkpoint(
                    latest_path,
                    epoch=epoch,
                    batch_index=batch_index,
                    best_accuracy_percent=best_accuracy,
                    extra={"learning_rate": learning_rate, "pause_boundary": True},
                )
                atomic_json(status_path, {
                    "status": "paused",
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "optimizer_step": state.optimizer_step,
                    "latest_checkpoint": str(latest_path),
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                raise SystemExit(0)
            augmented = augment_cifar10(
                raw_images, arguments.seed + epoch * 10_000_019 + batch_index)
            images = prepare_input(augmented, "raw-u8")
            batch_started = time.perf_counter()
            result = trainer.train_step(
                images,
                labels,
                learning_rate=learning_rate,
                momentum=arguments.momentum,
                weight_decay=arguments.weight_decay,
                gradient_clip_q15=arguments.gradient_clip_q15,
            )
            batch_seconds = time.perf_counter() - batch_started
            epoch_loss += result.loss * len(labels)
            epoch_correct += int(np.count_nonzero(result.predicted_class == labels))
            epoch_samples += len(labels)
            last_batch = batch_index
            if state.optimizer_step % arguments.checkpoint_interval == 0:
                checkpoint_hash = state.save_checkpoint(
                    latest_path,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    best_accuracy_percent=best_accuracy,
                    extra={"learning_rate": learning_rate},
                )
            else:
                checkpoint_hash = None
            event = {
                "event": "train_batch",
                "utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "batch_index": batch_index,
                "optimizer_step": result.optimizer_step,
                "samples": len(labels),
                "loss": result.loss,
                "batch_accuracy_percent": 100.0 * float(np.mean(result.predicted_class == labels)),
                "learning_rate": learning_rate,
                "elapsed_seconds": batch_seconds,
                "checkpoint_sha256": checkpoint_hash,
            }
            with history_path.open("a", encoding="ascii") as history:
                history.write(json.dumps(event, separators=(",", ":")) + "\n")
                history.flush()
                os.fsync(history.fileno())
            atomic_json(status_path, {
                "status": "running",
                "epoch": epoch,
                "epochs": arguments.epochs,
                "batch_index": batch_index + 1,
                "optimizer_step": state.optimizer_step,
                "running_loss": epoch_loss / epoch_samples,
                "running_accuracy_percent": 100.0 * epoch_correct / epoch_samples,
                "best_accuracy_percent": best_accuracy,
                "last_batch_seconds": batch_seconds,
                "latest_checkpoint": str(latest_path),
                "pause_file": str(pause_path),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })

        evaluation_accuracy, evaluation_correct, evaluation_total = evaluate(
            inference_model,
            test_data,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
        )
        improved = evaluation_accuracy > best_accuracy
        if improved:
            best_accuracy = evaluation_accuracy
        state.save_checkpoint(
            latest_path,
            epoch=epoch + 1,
            batch_index=0,
            best_accuracy_percent=best_accuracy,
            extra={"learning_rate": learning_rate},
        )
        if improved:
            temporary_best = best_path.with_name(best_path.name + ".tmp")
            shutil.copyfile(latest_path, temporary_best)
            os.replace(temporary_best, best_path)
        epoch_event = {
            "event": "epoch_complete",
            "utc": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "last_batch_index": last_batch,
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "train_accuracy_percent": 100.0 * epoch_correct / max(epoch_samples, 1),
            "evaluation_accuracy_percent": evaluation_accuracy,
            "evaluation_correct": evaluation_correct,
            "evaluation_samples": evaluation_total,
            "best_accuracy_percent": best_accuracy,
            "optimizer_step": state.optimizer_step,
            "performance_counters": asdict(driver.performance_snapshot()),
        }
        with history_path.open("a", encoding="ascii") as history:
            history.write(json.dumps(epoch_event, separators=(",", ":")) + "\n")
            history.flush()
            os.fsync(history.fileno())
        start_batch = 0

    atomic_json(status_path, {
        "status": "complete",
        "epochs": arguments.epochs,
        "optimizer_step": state.optimizer_step,
        "best_accuracy_percent": best_accuracy,
        "elapsed_seconds": time.time() - job_started,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
