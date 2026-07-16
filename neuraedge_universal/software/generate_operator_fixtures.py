"""Generate deterministic cross-language fixtures for vector and optimizer RTL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from math import isqrt

from neuraedge_golden import (
    VectorOperation,
    integer_layer_norm,
    log_int_softmax_q15,
    sgd_momentum_step,
    vector_alu,
)


def pack_signed(values: np.ndarray, bits: int) -> int:
    packed = 0
    mask = (1 << bits) - 1
    for lane, value in enumerate(np.asarray(values, dtype=np.int64).reshape(-1)):
        packed |= (int(value) & mask) << (lane * bits)
    return packed


def write_hex(path: Path, values: list[int], width_bits: int) -> None:
    digits = (width_bits + 3) // 4
    path.write_text("".join(f"{value & ((1 << width_bits) - 1):0{digits}x}\n" for value in values))


def generate_vector_cases(rng: np.random.Generator, count_per_operation: int) -> dict[str, list[int]]:
    commands: list[int] = []
    operands_a: list[int] = []
    operands_b: list[int] = []
    expected: list[int] = []
    boundaries = np.array(
        [-(1 << 31), -32768, -129, -1, 0, 1, 32767, (1 << 31) - 1], dtype=np.int64
    )

    for operation in VectorOperation:
        for case_index in range(count_per_operation):
            if case_index == 0:
                a = boundaries.copy()
                b = boundaries[::-1].copy()
            else:
                a = rng.integers(-(1 << 20), 1 << 20, size=8, dtype=np.int64)
                b = rng.integers(-(1 << 20), 1 << 20, size=8, dtype=np.int64)

            multiplier = int(rng.integers(-32768, 32768))
            if operation == VectorOperation.FLEX_REQUANTIZE:
                right_shift = int(rng.integers(-15, 16))
            else:
                right_shift = int(rng.integers(0, 8))
            zero_point = int(rng.integers(-128, 128))
            threshold = int(rng.integers(1, 4097))
            window = int(rng.integers(0, 4097))
            clamp_minimum = int(rng.integers(-32768, 0))
            clamp_maximum = int(rng.integers(0, 32768))
            result = vector_alu(
                operation,
                a,
                b,
                multiplier_q15=multiplier,
                right_shift=right_shift,
                output_zero_point=zero_point,
                threshold=threshold,
                surrogate_window=window,
                clamp_minimum=clamp_minimum,
                clamp_maximum=clamp_maximum,
            )

            command = 0
            command |= int(operation)
            command |= (multiplier & 0xFFFF) << 4
            command |= (right_shift & 0x1F) << 20
            command |= (zero_point & 0xFFFF) << 25
            command |= (threshold & 0xFFFFFFFF) << 41
            command |= window << 73
            command |= (clamp_minimum & 0xFFFFFFFF) << 105
            command |= (clamp_maximum & 0xFFFFFFFF) << 137
            commands.append(command)
            operands_a.append(pack_signed(a, 32))
            operands_b.append(pack_signed(b, 32))
            expected.append(pack_signed(result, 32))
    return {"commands": commands, "a": operands_a, "b": operands_b, "expected": expected}


def generate_optimizer_cases(rng: np.random.Generator, case_count: int) -> dict[str, list[int]]:
    parameters: list[int] = []
    weights: list[int] = []
    gradients: list[int] = []
    velocities: list[int] = []
    expected_weights: list[int] = []
    expected_velocities: list[int] = []

    for case_index in range(case_count):
        if case_index == 0:
            weight = np.array([-32768, -32767, -1, 0, 1, 32766, 32767, 123], dtype=np.int64)
            gradient = np.array([-(1 << 23), -(1 << 20), -1, 0, 1, 1 << 20, (1 << 23) - 1, -999])
            velocity = gradient[::-1].copy()
        elif case_index == 1:
            # This case distinguishes SGDW from coupled L2: decay changes the
            # weight direction but must not be retained in the velocity.
            weight = np.full(8, 1000, dtype=np.int64)
            gradient = np.zeros(8, dtype=np.int64)
            velocity = np.full(8, 1000, dtype=np.int64)
        else:
            weight = rng.integers(-32768, 32768, size=8, dtype=np.int64)
            gradient = rng.integers(-(1 << 23), 1 << 23, size=8, dtype=np.int64)
            velocity = rng.integers(-(1 << 23), 1 << 23, size=8, dtype=np.int64)
        if case_index == 1:
            learning_rate = 65536
            momentum = 65536
            weight_decay = 65536
            update_right_shift = 16
        else:
            learning_rate = int(rng.integers(1, 131072))
            momentum = int(rng.integers(0, 65537))
            weight_decay = int(rng.integers(0, 65537))
            update_right_shift = 16 if case_index == 0 else int(rng.integers(0, 32))
        next_weight, next_velocity = sgd_momentum_step(
            weight,
            gradient,
            velocity,
            learning_rate,
            momentum,
            weight_decay,
            update_right_shift,
        )
        parameters.append(
            learning_rate
            | (momentum << 18)
            | (weight_decay << 35)
            | (update_right_shift << 52)
        )
        weights.append(pack_signed(weight, 16))
        gradients.append(pack_signed(gradient, 24))
        velocities.append(pack_signed(velocity, 24))
        expected_weights.append(pack_signed(next_weight, 16))
        expected_velocities.append(pack_signed(next_velocity, 24))
    return {
        "parameters": parameters,
        "weights": weights,
        "gradients": gradients,
        "velocities": velocities,
        "expected_weights": expected_weights,
        "expected_velocities": expected_velocities,
    }


def generate_layernorm_cases(rng: np.random.Generator) -> dict[str, list[int]]:
    descriptors: list[int] = []
    values: list[int] = []
    gammas: list[int] = []
    betas: list[int] = []
    expected: list[int] = []
    debug: list[int] = []
    lengths = [1, 4, 7, 32, 64, 127, 384, 1536]

    for case_index, length in enumerate(lengths):
        epsilon = 1 + case_index * 3
        if case_index == 0:
            x = np.array([17], dtype=np.int64)
        elif case_index == 1:
            x = np.full(length, -91, dtype=np.int64)
        else:
            x = rng.integers(-8192, 8193, size=length, dtype=np.int64)
        gamma = rng.integers(8192, 49153, size=length, dtype=np.int64)
        beta = rng.integers(-2048, 2049, size=length, dtype=np.int64)
        output = integer_layer_norm(x, gamma, beta, epsilon)
        mean = int(x.sum() // length)
        centered = x - mean
        variance = int((centered * centered).sum() // length)
        sigma = max(1, isqrt(variance + epsilon))
        reciprocal = ((1 << 31) - 1) // sigma

        descriptors.append(length | (epsilon << 12))
        values.extend(int(value) for value in x)
        gammas.extend(int(value) for value in gamma)
        betas.extend(int(value) for value in beta)
        expected.extend(int(value) for value in output)
        debug.append(
            (mean & 0xFFFFFFFF)
            | (variance << 32)
            | (sigma << 64)
            | (reciprocal << 96)
        )
    return {
        "descriptors": descriptors,
        "values": values,
        "gammas": gammas,
        "betas": betas,
        "expected": expected,
        "debug": debug,
        "lengths": lengths,
    }


def generate_softmax_cases(rng: np.random.Generator) -> dict[str, list[int]]:
    descriptors: list[int] = []
    scores: list[int] = []
    expected: list[int] = []
    debug: list[int] = []
    lengths = [1, 4, 7, 32, 64, 127, 256]

    for case_index, length in enumerate(lengths):
        difference_shift = case_index % 8
        if case_index == 0:
            score = np.array([123], dtype=np.int64)
        else:
            score = rng.integers(-(1 << 18), 1 << 18, size=length, dtype=np.int64)
        probability = log_int_softmax_q15(score, difference_shift)
        difference = score.max() - score
        exponent = np.minimum(difference >> difference_shift, 15)
        unnormalized = 1 << (15 - exponent)
        denominator = int(unnormalized.sum())
        reciprocal = ((1 << 31) - 1) // denominator

        descriptors.append(length | (difference_shift << 9))
        scores.extend(int(value) for value in score)
        expected.extend(int(value) for value in probability)
        debug.append((int(score.max()) & 0xFFFFFF) | (denominator << 24) | (reciprocal << 48))
    return {
        "descriptors": descriptors,
        "scores": scores,
        "expected": expected,
        "debug": debug,
        "lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("tb/fixtures"))
    parser.add_argument("--vector-count-per-operation", type=int, default=24)
    parser.add_argument("--optimizer-cases", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260711)

    vector = generate_vector_cases(rng, args.vector_count_per_operation)
    optimizer = generate_optimizer_cases(rng, args.optimizer_cases)
    layernorm = generate_layernorm_cases(rng)
    softmax = generate_softmax_cases(rng)
    write_hex(args.output_dir / "vector_commands.mem", vector["commands"], 192)
    write_hex(args.output_dir / "vector_a.mem", vector["a"], 256)
    write_hex(args.output_dir / "vector_b.mem", vector["b"], 256)
    write_hex(args.output_dir / "vector_expected.mem", vector["expected"], 256)
    write_hex(args.output_dir / "optimizer_parameters.mem", optimizer["parameters"], 64)
    write_hex(args.output_dir / "optimizer_weights.mem", optimizer["weights"], 128)
    write_hex(args.output_dir / "optimizer_gradients.mem", optimizer["gradients"], 192)
    write_hex(args.output_dir / "optimizer_velocities.mem", optimizer["velocities"], 192)
    write_hex(args.output_dir / "optimizer_expected_weights.mem", optimizer["expected_weights"], 128)
    write_hex(args.output_dir / "optimizer_expected_velocities.mem", optimizer["expected_velocities"], 192)
    write_hex(args.output_dir / "layernorm_cases.mem", layernorm["descriptors"], 64)
    write_hex(args.output_dir / "layernorm_values.mem", layernorm["values"], 24)
    write_hex(args.output_dir / "layernorm_gammas.mem", layernorm["gammas"], 24)
    write_hex(args.output_dir / "layernorm_betas.mem", layernorm["betas"], 24)
    write_hex(args.output_dir / "layernorm_expected.mem", layernorm["expected"], 32)
    write_hex(args.output_dir / "layernorm_debug.mem", layernorm["debug"], 128)
    write_hex(args.output_dir / "softmax_cases.mem", softmax["descriptors"], 32)
    write_hex(args.output_dir / "softmax_scores.mem", softmax["scores"], 24)
    write_hex(args.output_dir / "softmax_expected.mem", softmax["expected"], 16)
    write_hex(args.output_dir / "softmax_debug.mem", softmax["debug"], 80)

    manifest = {
        "seed": 20260711,
        "vector_cases": len(vector["commands"]),
        "vector_cases_per_operation": args.vector_count_per_operation,
        "optimizer_cases": len(optimizer["parameters"]),
        "layernorm_cases": len(layernorm["lengths"]),
        "layernorm_samples": len(layernorm["values"]),
        "softmax_cases": len(softmax["lengths"]),
        "softmax_samples": len(softmax["scores"]),
        "rounding": "nearest, ties away from zero",
    }
    (args.output_dir / "operator_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_dir / "operator_fixture_params.svh").write_text(
        f"`define NEURAEDGE_VECTOR_CASES {manifest['vector_cases']}\n"
        f"`define NEURAEDGE_OPTIMIZER_CASES {manifest['optimizer_cases']}\n"
        f"`define NEURAEDGE_LAYERNORM_CASES {manifest['layernorm_cases']}\n"
        f"`define NEURAEDGE_LAYERNORM_SAMPLES {manifest['layernorm_samples']}\n"
        f"`define NEURAEDGE_SOFTMAX_CASES {manifest['softmax_cases']}\n"
        f"`define NEURAEDGE_SOFTMAX_SAMPLES {manifest['softmax_samples']}\n"
    )


if __name__ == "__main__":
    main()
