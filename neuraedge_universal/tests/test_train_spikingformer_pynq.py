from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_spikingformer_pynq import (
    Cifar10ShardSet,
    augment_cifar10,
    cosine_learning_rate,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PynqTrainingJobTests(unittest.TestCase):
    def test_shards_are_integrity_checked_deterministic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            first_index = 0
            for shard_index, samples in enumerate((5, 4)):
                path = root / f"cifar10_train_{shard_index:03d}.npz"
                images = np.full((samples, 32, 32, 3), shard_index * 40, dtype=np.uint8)
                labels = np.arange(first_index, first_index + samples, dtype=np.uint8) % 10
                np.savez_compressed(path, images=images, labels=labels)
                entries.append({
                    "file": path.name,
                    "samples": samples,
                    "first_index": first_index,
                    "last_index_exclusive": first_index + samples,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
                first_index += samples
            manifest = {
                "format": "neuraedge-official-cifar10-shards-v1",
                "splits": {"train": entries, "test": entries},
            }
            manifest_path = root / "cifar10_shards_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            dataset = Cifar10ShardSet(manifest_path, "train")
            first = list(dataset.batches(
                epoch=7, batch_size=2, seed=19, shuffle=True))
            second = list(dataset.batches(
                epoch=7, batch_size=2, seed=19, shuffle=True))
            self.assertEqual([item[0] for item in first], list(range(len(first))))
            self.assertEqual(
                [item[2].tolist() for item in first],
                [item[2].tolist() for item in second],
            )
            resumed = list(dataset.batches(
                epoch=7, batch_size=2, seed=19, shuffle=True, start_batch=3))
            self.assertEqual([item[0] for item in resumed], list(range(3, len(first))))
            self.assertEqual(sum(len(item[2]) for item in first), 9)

            manifest["splits"]["train"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="ascii")
            with self.assertRaises(ValueError):
                Cifar10ShardSet(manifest_path, "train")

    def test_augmentation_and_cosine_schedule_are_reproducible(self) -> None:
        images = np.arange(2 * 32 * 32 * 3, dtype=np.uint16).reshape(2, 32, 32, 3)
        images = (images % 256).astype(np.uint8)
        first = augment_cifar10(images, 1234)
        second = augment_cifar10(images, 1234)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, images.shape)
        self.assertEqual(cosine_learning_rate(0, 400, 0.25, 0.01), 0.25)
        self.assertAlmostEqual(cosine_learning_rate(399, 400, 0.25, 0.01), 0.01)


if __name__ == "__main__":
    unittest.main()
