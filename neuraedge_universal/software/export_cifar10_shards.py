"""Download the official CIFAR-10 split and export board-friendly NPZ shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_split(
    dataset,
    split: str,
    output_dir: Path,
    shard_size: int,
) -> list[dict[str, object]]:
    images = np.asarray(dataset.data, dtype=np.uint8)
    labels = np.asarray(dataset.targets, dtype=np.uint8)
    if images.shape != (len(labels), 32, 32, 3):
        raise ValueError(f"unexpected official CIFAR-10 {split} shape {images.shape}")
    records: list[dict[str, object]] = []
    for shard_index, start in enumerate(range(0, len(labels), shard_size)):
        stop = min(start + shard_size, len(labels))
        target = output_dir / f"cifar10_{split}_{shard_index:03d}.npz"
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, images=images[start:stop], labels=labels[start:stop])
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        records.append({
            "file": target.name,
            "samples": stop - start,
            "first_index": start,
            "last_index_exclusive": stop,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })
    return records


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as error:
        raise RuntimeError("torchvision is required to export official CIFAR-10") from error
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = arguments.dataset_root.expanduser().resolve()
    train = CIFAR10(
        root=str(dataset_root), train=True, download=not arguments.no_download)
    test = CIFAR10(
        root=str(dataset_root), train=False, download=not arguments.no_download)
    manifest = {
        "format": "neuraedge-official-cifar10-shards-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "torchvision.datasets.CIFAR10 official Python archive",
        "image_encoding": "uint8 NHWC RGB",
        "label_encoding": "uint8 class index 0..9",
        "shard_size": arguments.shard_size,
        "splits": {
            "train": write_split(train, "train", output_dir, arguments.shard_size),
            "test": write_split(test, "test", output_dir, arguments.shard_size),
        },
    }
    manifest_path = output_dir / "cifar10_shards_manifest.json"
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({
        "manifest": str(manifest_path),
        "train_samples": sum(item["samples"] for item in manifest["splits"]["train"]),
        "test_samples": sum(item["samples"] for item in manifest["splits"]["test"]),
    }, indent=2))
