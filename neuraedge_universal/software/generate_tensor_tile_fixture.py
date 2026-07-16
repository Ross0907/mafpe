"""Generate deterministic, independently computed tensor-tile RTL fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuraedge_golden import (
    ComputeMode,
    binary_weighted_gemm,
    gemm_u8_zero_point,
    ternary_weighted_gemm,
)


ROW_LANES = 8
COLUMN_LANES = 16


@dataclass
class FixtureCase:
    name: str
    mode: ComputeMode
    rows: int
    columns: int
    activation_zero_point: int
    weight_zero_point: int
    activation_words: list[int]
    weight_words: list[int]
    expected: np.ndarray


def pack_bytes(values: np.ndarray | list[int]) -> int:
    packed = 0
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        packed |= (int(value) & 0xFF) << (8 * index)
    return packed


def make_dense_outer(rng: np.random.Generator) -> FixtureCase:
    rows, columns, reduction = 8, 16, 19
    za, zb = 117, 139
    a = rng.integers(0, 256, size=(rows, reduction), dtype=np.uint8)
    b = rng.integers(0, 256, size=(reduction, columns), dtype=np.uint8)
    a[:, [2, 7, 14]] = za
    expected = gemm_u8_zero_point(a, b, za, zb)
    a_words = [pack_bytes(a[:, k]) for k in range(reduction)]
    b_words = [pack_bytes(b[k]) for k in range(reduction)]
    return FixtureCase("dense_outer_8x16x19", ComputeMode.DENSE_OUTER, rows, columns, za, zb, a_words, b_words, expected)


def make_dense_dot8(rng: np.random.Generator) -> FixtureCase:
    rows, columns, reduction = 1, 13, 21
    padded_reduction = 24
    za, zb = 128, 128
    a = np.full((rows, padded_reduction), za, dtype=np.uint8)
    b = np.full((padded_reduction, columns), zb, dtype=np.uint8)
    a[:, :reduction] = rng.integers(0, 256, size=(rows, reduction), dtype=np.uint8)
    b[:reduction] = rng.integers(0, 256, size=(reduction, columns), dtype=np.uint8)
    a[:, 8:16] = za
    expected = gemm_u8_zero_point(a, b, za, zb)
    a_words: list[int] = []
    b_words: list[int] = []
    for chunk_start in range(0, padded_reduction, ROW_LANES):
        a_words.append(pack_bytes(a[0, chunk_start : chunk_start + ROW_LANES]))
        packed_weights = 0
        for column in range(COLUMN_LANES):
            lane_values = (
                b[chunk_start : chunk_start + ROW_LANES, column]
                if column < columns
                else np.full(ROW_LANES, zb, dtype=np.uint8)
            )
            packed_weights |= pack_bytes(lane_values) << (64 * column)
        b_words.append(packed_weights)
    return FixtureCase("dense_dot8_1x13x24", ComputeMode.DENSE_DOT8, rows, columns, za, zb, a_words, b_words, expected)


def make_binary(rng: np.random.Generator) -> FixtureCase:
    rows, columns, reduction = 5, 11, 17
    spikes = rng.integers(0, 2, size=(rows, reduction), dtype=np.int8)
    weights = rng.integers(-128, 128, size=(reduction, columns), dtype=np.int16)
    spikes[:, [4, 12]] = 0
    expected = binary_weighted_gemm(spikes, weights)
    a_words: list[int] = []
    b_words: list[int] = []
    for k in range(reduction):
        a_padded = np.zeros(ROW_LANES, dtype=np.int16)
        b_padded = np.zeros(COLUMN_LANES, dtype=np.int16)
        a_padded[:rows] = spikes[:, k]
        b_padded[:columns] = weights[k]
        a_words.append(pack_bytes(a_padded))
        b_words.append(pack_bytes(b_padded))
    return FixtureCase("binary_weighted_5x11x17", ComputeMode.BINARY_WEIGHTED, rows, columns, 0, 0, a_words, b_words, expected)


def make_ternary(rng: np.random.Generator) -> FixtureCase:
    rows, columns, reduction = 7, 9, 15
    spikes = rng.integers(-1, 2, size=(rows, reduction), dtype=np.int8)
    weights = rng.integers(-128, 128, size=(reduction, columns), dtype=np.int16)
    spikes[:, [1, 10]] = 0
    expected = ternary_weighted_gemm(spikes, weights)
    a_words: list[int] = []
    b_words: list[int] = []
    for k in range(reduction):
        a_padded = np.zeros(ROW_LANES, dtype=np.int16)
        b_padded = np.zeros(COLUMN_LANES, dtype=np.int16)
        a_padded[:rows] = spikes[:, k]
        b_padded[:columns] = weights[k]
        a_words.append(pack_bytes(a_padded))
        b_words.append(pack_bytes(b_padded))
    return FixtureCase("ternary_weighted_7x9x15", ComputeMode.TERNARY_WEIGHTED, rows, columns, 0, 0, a_words, b_words, expected)


def command_word(case: FixtureCase) -> int:
    issues = len(case.activation_words)
    return (
        int(case.mode)
        | (case.rows << 4)
        | (case.columns << 8)
        | (issues << 14)
        | (case.activation_zero_point << 26)
        | (case.weight_zero_point << 34)
    )


def write_hex(path: Path, values: list[int], width_bits: int) -> None:
    digits = (width_bits + 3) // 4
    mask = (1 << width_bits) - 1
    path.write_text("".join(f"{value & mask:0{digits}x}\n" for value in values), encoding="ascii")


def main() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "tb" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0x4E455552)
    cases = [make_dense_outer(rng), make_dense_dot8(rng), make_binary(rng), make_ternary(rng)]

    commands = [command_word(case) for case in cases]
    activation_words = [word for case in cases for word in case.activation_words]
    weight_words = [word for case in cases for word in case.weight_words]
    expected_values = [int(value) for case in cases for value in case.expected.reshape(-1)]

    write_hex(fixture_dir / "tensor_cases.mem", commands, 64)
    write_hex(fixture_dir / "tensor_activations.mem", activation_words, 64)
    write_hex(fixture_dir / "tensor_weights.mem", weight_words, COLUMN_LANES * ROW_LANES * 8)
    write_hex(fixture_dir / "tensor_expected.mem", expected_values, 64)
    (fixture_dir / "tensor_tile_params.svh").write_text(
        "`define NEURAEDGE_TENSOR_CASES %d\n"
        "`define NEURAEDGE_TENSOR_ISSUES %d\n"
        "`define NEURAEDGE_TENSOR_RESULTS %d\n"
        "`define NEURAEDGE_COLUMN_LANES %d\n"
        % (len(cases), len(activation_words), len(expected_values), COLUMN_LANES),
        encoding="ascii",
    )

    manifest = {
        "seed": "0x4E455552",
        "case_count": len(cases),
        "issue_count": len(activation_words),
        "result_count": len(expected_values),
        "cases": [
            {
                "name": case.name,
                "mode": case.mode.name,
                "rows": case.rows,
                "columns": case.columns,
                "issues": len(case.activation_words),
                "activation_zero_point": case.activation_zero_point,
                "weight_zero_point": case.weight_zero_point,
                "skipped_zero_issues": sum(word == pack_bytes([case.activation_zero_point] * ROW_LANES) for word in case.activation_words)
                if case.mode in (ComputeMode.DENSE_OUTER, ComputeMode.DENSE_DOT8)
                else sum(word == 0 for word in case.activation_words),
            }
            for case in cases
        ],
    }
    (fixture_dir / "tensor_fixture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
