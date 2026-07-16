"""Strict reader for the exported NeuraEdge Spikingformer parameter archive."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FORMAT = "neuraedge-spikingformer-parameters-v1"


@dataclass(frozen=True)
class ProjectionParameters:
    """Integer tensors consumed by one folded Conv-BN hardware projection."""

    name: str
    weight_k_by_n: np.ndarray
    bias_q10: np.ndarray
    multiplier_q15: np.ndarray
    binary_point_shift: np.ndarray
    compute_opcode: str
    source_weight_shape: tuple[int, ...]
    output_fractional_bits: int


@dataclass(frozen=True)
class HeadParameters:
    """Integer classifier matrix and bias in accumulator units."""

    weight_k_by_n: np.ndarray
    bias_accumulator_units: np.ndarray


class SpikingformerParameterArchive:
    """Validated, read-only view of one binary parameter package."""

    def __init__(self, manifest_path: str | Path, *, verify_hashes: bool = True) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        manifest_object = json.loads(self.manifest_path.read_text(encoding="ascii"))
        if not isinstance(manifest_object, Mapping):
            raise ValueError("parameter manifest root must be an object")
        self.manifest: Mapping[str, Any] = manifest_object
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"unsupported parameter format {self.manifest.get('format')!r}")

        binary_name = self.manifest.get("binary_file")
        if not isinstance(binary_name, str) or Path(binary_name).name != binary_name:
            raise ValueError("binary_file must be a file name beside the manifest")
        self.binary_path = self.manifest_path.parent / binary_name
        self._binary = self.binary_path.read_bytes()
        declared_bytes = self._integer_field(self.manifest, "binary_bytes")
        if len(self._binary) != declared_bytes:
            raise ValueError(
                f"parameter binary has {len(self._binary)} bytes; expected {declared_bytes}")
        if verify_hashes:
            self._verify_hash(self._binary, self.manifest.get("binary_sha256"), "binary")

        alignment = self._integer_field(self.manifest, "alignment_bytes")
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment_bytes must be a positive power of two")
        section_objects = self.manifest.get("sections")
        if not isinstance(section_objects, list) or not section_objects:
            raise ValueError("manifest must contain a nonempty sections list")
        self._sections: dict[str, np.ndarray] = {}
        occupied_end = 0
        for section_object in section_objects:
            if not isinstance(section_object, Mapping):
                raise ValueError("each section must be an object")
            name = section_object.get("name")
            if not isinstance(name, str) or not name or name in self._sections:
                raise ValueError(f"invalid or duplicate section name {name!r}")
            offset = self._integer_field(section_object, "offset")
            length = self._integer_field(section_object, "length")
            if offset % alignment:
                raise ValueError(f"section {name} offset {offset} is not {alignment}-byte aligned")
            if offset < occupied_end:
                raise ValueError(f"section {name} overlaps a preceding section")
            if length < 0 or offset + length > len(self._binary):
                raise ValueError(f"section {name} is outside the parameter binary")
            dtype_value = section_object.get("dtype")
            shape_value = section_object.get("shape")
            if not isinstance(dtype_value, str) or not isinstance(shape_value, list):
                raise ValueError(f"section {name} has invalid dtype or shape metadata")
            dtype = np.dtype(dtype_value)
            shape = tuple(self._nonnegative_dimension(value, name) for value in shape_value)
            expected_length = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if length != expected_length:
                raise ValueError(
                    f"section {name} has {length} bytes; dtype/shape require {expected_length}")
            payload = self._binary[offset:offset + length]
            if verify_hashes:
                self._verify_hash(payload, section_object.get("sha256"), f"section {name}")
            array = np.frombuffer(payload, dtype=dtype).reshape(shape)
            array.flags.writeable = False
            self._sections[name] = array
            occupied_end = offset + length

        declared_section_names = set(self._sections)
        referenced_section_names: set[str] = set()
        for projection in self._projection_objects():
            referenced_section_names.update(self._section_names(projection, "projection"))
        head = self.manifest.get("head")
        if not isinstance(head, Mapping):
            raise ValueError("manifest head must be an object")
        referenced_section_names.update(self._section_names(head, "head"))
        if referenced_section_names != declared_section_names:
            missing = sorted(declared_section_names - referenced_section_names)
            unknown = sorted(referenced_section_names - declared_section_names)
            raise ValueError(f"section reference mismatch: unreferenced={missing}, unknown={unknown}")

    @staticmethod
    def _integer_field(mapping: Mapping[str, Any], name: str) -> int:
        value = mapping.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @staticmethod
    def _nonnegative_dimension(value: Any, section_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"section {section_name} has an invalid shape dimension")
        return value

    @staticmethod
    def _verify_hash(payload: bytes, expected: Any, description: str) -> None:
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{description} SHA-256 metadata is invalid")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected.lower():
            raise ValueError(f"{description} SHA-256 mismatch: {actual} != {expected}")

    def _projection_objects(self) -> list[Mapping[str, Any]]:
        projections = self.manifest.get("projections")
        if not isinstance(projections, list):
            raise ValueError("manifest projections must be a list")
        if not all(isinstance(item, Mapping) for item in projections):
            raise ValueError("each projection must be an object")
        return projections

    @staticmethod
    def _section_names(owner: Mapping[str, Any], owner_kind: str) -> list[str]:
        sections = owner.get("sections")
        if not isinstance(sections, list):
            raise ValueError(f"{owner_kind} sections must be a list")
        names: list[str] = []
        for section in sections:
            if not isinstance(section, Mapping) or not isinstance(section.get("name"), str):
                raise ValueError(f"{owner_kind} has an invalid section reference")
            names.append(section["name"])
        return names

    def section(self, name: str) -> np.ndarray:
        try:
            return self._sections[name]
        except KeyError as error:
            raise KeyError(f"parameter section {name!r} does not exist") from error

    @property
    def projection_names(self) -> tuple[str, ...]:
        return tuple(str(item.get("name")) for item in self._projection_objects())

    def projection(self, name: str) -> ProjectionParameters:
        matches = [item for item in self._projection_objects() if item.get("name") == name]
        if len(matches) != 1:
            raise KeyError(f"projection {name!r} does not exist uniquely")
        owner = matches[0]
        section_names = self._section_names(owner, f"projection {name}")
        expected_suffixes = (
            ".weight_k_by_n_int8", ".bias_q10", ".multiplier_q15",
            ".binary_point_shift",
        )
        selected: list[np.ndarray] = []
        for suffix in expected_suffixes:
            suffix_matches = [entry for entry in section_names if entry.endswith(suffix)]
            if len(suffix_matches) != 1:
                raise ValueError(f"projection {name} must reference one {suffix} section")
            selected.append(self.section(suffix_matches[0]))
        weight, bias, multiplier, shift = selected
        if weight.ndim != 2 or any(value.ndim != 1 for value in selected[1:]):
            raise ValueError(f"projection {name} section ranks are invalid")
        if not (weight.shape[1] == len(bias) == len(multiplier) == len(shift)):
            raise ValueError(f"projection {name} output-channel metadata is inconsistent")
        source_shape_object = owner.get("source_weight_shape")
        if not isinstance(source_shape_object, list):
            raise ValueError(f"projection {name} has no source weight shape")
        source_shape = tuple(
            self._nonnegative_dimension(value, f"projection {name}")
            for value in source_shape_object)
        if len(source_shape) not in (3, 4) or source_shape[0] != weight.shape[1]:
            raise ValueError(f"projection {name} source weight shape is inconsistent")
        fractional_bits = self._integer_field(owner, "output_fractional_bits")
        return ProjectionParameters(
            name, weight, bias, multiplier, shift, str(owner.get("compute_opcode")),
            source_shape, fractional_bits)

    @property
    def head(self) -> HeadParameters:
        owner = self.manifest["head"]
        names = self._section_names(owner, "head")
        weight_names = [name for name in names if name.endswith(".weight_k_by_n_int8")]
        bias_names = [name for name in names if name.endswith(".bias_accumulator_units")]
        if len(weight_names) != 1 or len(bias_names) != 1:
            raise ValueError("head must reference one weight and one bias section")
        weight = self.section(weight_names[0])
        bias = self.section(bias_names[0])
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[1] != len(bias):
            raise ValueError("head section shapes are inconsistent")
        return HeadParameters(weight, bias)
