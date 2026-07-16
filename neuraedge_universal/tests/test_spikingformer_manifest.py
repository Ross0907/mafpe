from __future__ import annotations

import unittest

from generate_spikingformer_manifest import build_operator_manifest, matrix_schedule


class MatrixScheduleTests(unittest.TestCase):
    def test_outer_product_schedule_counts_tiles_and_commands(self) -> None:
        schedule = matrix_schedule(
            9, 17, 18, arithmetic="binary_spike_add")
        self.assertEqual(schedule.compute_tiles, 4)
        self.assertEqual(schedule.commands, 10)
        self.assertEqual(schedule.mac_equivalents, 9 * 17 * 18)
        self.assertEqual(schedule.reduction_chunks, 1)

    def test_dot8_schedule_pads_reduction_without_changing_mac_count(self) -> None:
        schedule = matrix_schedule(1, 17, 18, arithmetic="u8_mac")
        self.assertEqual(schedule.compute_tiles, 2)
        self.assertEqual(schedule.commands, 5)
        self.assertEqual(schedule.mac_equivalents, 1 * 17 * 18)

    def test_large_reduction_is_split_at_hardware_limit(self) -> None:
        schedule = matrix_schedule(8, 3456, 16, arithmetic="binary_spike_add")
        self.assertEqual(schedule.reduction_chunks, 2)
        self.assertEqual(schedule.compute_tiles, 2)
        self.assertEqual(schedule.commands, 6)


class SpikingformerManifestTests(unittest.TestCase):
    def test_manifest_covers_every_learned_projection(self) -> None:
        operators = build_operator_manifest()
        matrix_operators = [item for item in operators if item.matrix is not None]
        self.assertEqual(len(matrix_operators), 5 + 4 * 8 + 1)
        names = {item.name for item in matrix_operators}
        self.assertIn("patch_embed.block0", names)
        self.assertIn("block.3.attn.qk", names)
        self.assertIn("block.3.mlp.fc2", names)
        self.assertIn("head", names)

    def test_temporal_hoist_counts_block0_once(self) -> None:
        operators = {item.name: item for item in build_operator_manifest()}
        block0 = operators["patch_embed.block0"]
        self.assertEqual(block0.matrix.repetitions, 1)
        self.assertIsNotNone(block0.exact_optimization)


if __name__ == "__main__":
    unittest.main()
