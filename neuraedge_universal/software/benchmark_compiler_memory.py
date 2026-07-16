"""Measure compilation time and Python memory for one real exported projection."""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuraedge_compiler import compile_spike_gemm
from spikingformer_parameters import SpikingformerParameterArchive


def benchmark(manifest: Path, projection_name: str, rows: int) -> dict[str, object]:
    if rows <= 0:
        raise ValueError("rows must be positive")
    archive = SpikingformerParameterArchive(manifest)
    projection = archive.projection(projection_name)
    spikes = np.zeros((rows, projection.weight_k_by_n.shape[0]), dtype=np.uint8)
    tracemalloc.start()
    started = time.perf_counter()
    program = compile_spike_gemm(spikes, projection.weight_k_by_n)
    elapsed = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output_beats = sum(instruction.output_beats for instruction in program.instructions)
    descriptor_bytes = 24 * len(program.instructions)
    return {
        "format": "neuraedge-compiler-memory-benchmark-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "manifest": str(manifest.resolve()),
        "projection": projection_name,
        "shape": list(program.shape),
        "instructions": len(program.instructions),
        "payload_input_bytes": program.estimated_input_bytes,
        "descriptor_bytes": descriptor_bytes,
        "dma_input_bytes": program.estimated_input_bytes + descriptor_bytes,
        "dma_output_bytes": output_beats * 8,
        "compile_seconds": elapsed,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--projection", default="patch_embed.block3")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = benchmark(arguments.manifest, arguments.projection, arguments.rows)
    payload = json.dumps(result, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="ascii")
    print(payload, end="")
