from __future__ import annotations

import unittest

import numpy as np

from run_spikingformer_pynq import CIFAR10_MEAN, CIFAR10_STD, prepare_input


class PynqInferenceInputTests(unittest.TestCase):
    def test_raw_nhwc_and_nchw_normalization(self) -> None:
        raw_nhwc = np.zeros((32, 32, 3), dtype=np.uint8)
        raw_nhwc[..., 1] = 255
        result = prepare_input(raw_nhwc, "raw-u8")
        self.assertEqual(result.shape, (1, 3, 32, 32))
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), -float(CIFAR10_MEAN[0] / CIFAR10_STD[0]), places=6)
        self.assertAlmostEqual(float(result[0, 1, 0, 0]), float((1 - CIFAR10_MEAN[1]) / CIFAR10_STD[1]), places=6)
        np.testing.assert_array_equal(
            prepare_input(result, "normalized"), result.astype(np.float32))

    def test_rejects_wrong_shape_and_out_of_range_raw_values(self) -> None:
        with self.assertRaises(ValueError):
            prepare_input(np.zeros((28, 28), dtype=np.uint8), "raw-u8")
        with self.assertRaises(ValueError):
            prepare_input(np.full((3, 32, 32), 256), "raw-u8")


if __name__ == "__main__":
    unittest.main()
