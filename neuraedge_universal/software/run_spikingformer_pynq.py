"""Run complete trained Spikingformer inference on a PYNQ-Z2 overlay."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuraedge_pynq import NeuraEdgeDriver
from spikingformer_accelerator import SpikingformerAcceleratorModel
from spikingformer_parameters import SpikingformerParameterArchive


CIFAR10_MEAN = np.asarray((0.4914, 0.4822, 0.4465), dtype=np.float32)
CIFAR10_STD = np.asarray((0.2470, 0.2435, 0.2616), dtype=np.float32)


def prepare_input(value: np.ndarray, input_format: str) -> np.ndarray:
    """Return normalized NCHW float32 data from raw-u8 or normalized arrays."""

    array = np.asarray(value)
    if array.ndim == 3:
        array = array[np.newaxis]
    if array.ndim != 4:
        raise ValueError("input array must have three or four dimensions")
    if array.shape[1:] == (3, 32, 32):
        nchw = array
    elif array.shape[1:] == (32, 32, 3):
        nchw = array.transpose(0, 3, 1, 2)
    else:
        raise ValueError("input array must be NCHW or NHWC CIFAR-10 shape")
    if input_format == "normalized":
        return nchw.astype(np.float32)
    if input_format != "raw-u8":
        raise ValueError("input_format must be normalized or raw-u8")
    if not np.issubdtype(nchw.dtype, np.integer) or np.any(nchw < 0) or np.any(nchw > 255):
        raise ValueError("raw-u8 input must contain integer pixels in [0,255]")
    unit = nchw.astype(np.float32) / np.float32(255.0)
    return (unit - CIFAR10_MEAN.reshape(1, 3, 1, 1)) / CIFAR10_STD.reshape(1, 3, 1, 1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-format", choices=("normalized", "raw-u8"), default="raw-u8")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    archive = SpikingformerParameterArchive(arguments.manifest)
    image = prepare_input(np.load(arguments.input, allow_pickle=False), arguments.input_format)
    driver = NeuraEdgeDriver.from_bitfile(
        arguments.bitfile, download=not arguments.no_download)
    model = SpikingformerAcceleratorModel(
        archive, driver, layer_timeout_seconds=arguments.layer_timeout_seconds)
    started = time.perf_counter()
    inference = model.infer(image)
    elapsed = time.perf_counter() - started
    performance = driver.performance_snapshot()
    report = {
        "format": "neuraedge-pynq-spikingformer-inference-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bitfile": str(arguments.bitfile.resolve()),
        "parameter_manifest": str(arguments.manifest.resolve()),
        "parameter_binary_sha256": archive.manifest["binary_sha256"],
        "batch_size": int(image.shape[0]),
        "elapsed_seconds": elapsed,
        "images_per_second": image.shape[0] / elapsed,
        "predicted_class": inference.predicted_class.tolist(),
        "logits_accumulator": inference.logits_accumulator.tolist(),
        "logits": inference.logits.tolist(),
        "performance_counters": asdict(performance),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
