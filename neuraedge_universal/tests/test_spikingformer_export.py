from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch
import torch.nn as nn

from export_spikingformer_hardware import (
    ALIGNMENT,
    AlignedBinaryWriter,
    export_projection,
    quantize_weight_tensor,
)


class HardwareExportTests(unittest.TestCase):
    def test_aligned_writer_records_exact_payload_hash(self) -> None:
        writer = AlignedBinaryWriter()
        first = writer.append("first", np.arange(7, dtype=np.int8))
        second_value = np.arange(5, dtype="<i4")
        second = writer.append("second", second_value)
        self.assertEqual(first.offset, 0)
        self.assertEqual(second.offset % ALIGNMENT, 0)
        payload = second_value.tobytes()
        self.assertEqual(second.sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(bytes(writer.data[second.offset:second.offset + second.length]), payload)

    def test_tensor_weight_quantization_uses_complete_int8_range(self) -> None:
        weight = torch.tensor([[-2.0, 0.0, 1.0, 2.0]])
        integer, scale = quantize_weight_tensor(weight)
        self.assertEqual(scale, 2.0 / 127.0)
        torch.testing.assert_close(integer, torch.tensor([[-127, 0, 64, 127]], dtype=torch.int8))

    def test_projection_export_matches_folded_float_equation(self) -> None:
        torch.manual_seed(19)
        convolution = nn.Conv1d(4, 3, kernel_size=1, bias=True).eval()
        batch_norm = nn.BatchNorm1d(3).eval()
        batch_norm.running_mean.copy_(torch.randn(3))
        batch_norm.running_var.copy_(torch.rand(3) + 0.5)
        writer = AlignedBinaryWriter()
        exported = export_projection(
            writer,
            name="probe",
            convolution=convolution,
            batch_norm=batch_norm,
            input_scale=1.0,
            fractional_bits=10,
            input_encoding="binary spike 0/1",
            compute_opcode="COMPUTE_BINARY",
        )
        self.assertEqual(exported.hardware_matrix_shape, (4, 3))
        self.assertEqual(len(exported.sections), 4)
        self.assertLess(exported.represented_factor_maximum_error, 0.001)
        for section in exported.sections:
            self.assertEqual(section.offset % ALIGNMENT, 0)


if __name__ == "__main__":
    unittest.main()
