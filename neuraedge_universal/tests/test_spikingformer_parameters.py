from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_spikingformer_hardware import AlignedBinaryWriter
from spikingformer_parameters import FORMAT, SpikingformerParameterArchive


class ParameterArchiveTests(unittest.TestCase):
    def _write_archive(self, directory: Path) -> Path:
        writer = AlignedBinaryWriter()
        sections = [
            writer.append("unit.weight_k_by_n_int8", np.arange(12, dtype=np.int8).reshape(3, 4)),
            writer.append("unit.bias_q10", np.arange(4, dtype="<i4")),
            writer.append("unit.multiplier_q15", np.arange(4, dtype="<i2")),
            writer.append("unit.binary_point_shift", np.arange(4, dtype=np.int8)),
            writer.append("head.weight_k_by_n_int8", np.arange(8, dtype=np.int8).reshape(4, 2)),
            writer.append("head.bias_accumulator_units", np.arange(2, dtype="<i4")),
        ]
        binary = directory / "parameters.bin"
        binary.write_bytes(writer.data)
        manifest = {
            "format": FORMAT,
            "binary_file": binary.name,
            "binary_bytes": len(writer.data),
            "binary_sha256": hashlib.sha256(writer.data).hexdigest(),
            "alignment_bytes": 64,
            "projections": [{
                "name": "unit",
                "compute_opcode": "COMPUTE_BINARY",
                "source_weight_shape": [4, 3, 1, 1],
                "output_fractional_bits": 10,
                "sections": [section.__dict__ for section in sections[:4]],
            }],
            "head": {"sections": [section.__dict__ for section in sections[4:]]},
            "sections": [section.__dict__ for section in sections],
        }
        path = directory / "parameters.json"
        path.write_text(json.dumps(manifest), encoding="ascii")
        return path

    def test_loads_typed_read_only_projection_and_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = SpikingformerParameterArchive(self._write_archive(Path(temporary)))
            projection = archive.projection("unit")
            self.assertEqual(projection.weight_k_by_n.shape, (3, 4))
            self.assertEqual(projection.weight_k_by_n.dtype, np.dtype("int8"))
            self.assertFalse(projection.weight_k_by_n.flags.writeable)
            self.assertEqual(archive.head.weight_k_by_n.shape, (4, 2))
            self.assertEqual(archive.projection_names, ("unit",))

    def test_rejects_one_corrupted_parameter_byte(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self._write_archive(directory)
            binary = directory / "parameters.bin"
            payload = bytearray(binary.read_bytes())
            payload[0] ^= 0x80
            binary.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                SpikingformerParameterArchive(manifest)

    def test_rejects_overlapping_section_metadata_without_hash_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest_path = self._write_archive(directory)
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["sections"][1]["offset"] = manifest["sections"][0]["offset"]
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                SpikingformerParameterArchive(manifest_path, verify_hashes=False)


if __name__ == "__main__":
    unittest.main()
