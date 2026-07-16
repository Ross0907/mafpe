"""Static checks for board-build settings that simulations cannot observe."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PynqBuildContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = (ROOT / "scripts" / "build_pynq_full.tcl").read_text(
            encoding="ascii")

    def test_dma_accepts_a_complete_spikingformer_layer_program(self) -> None:
        match = re.search(r"CONFIG\.c_sg_length_width\s+\{(\d+)\}", self.script)
        self.assertIsNotNone(match, "AXI DMA length width must be explicit")
        self.assertEqual(int(match.group(1)), 26)
        self.assertGreater((1 << int(match.group(1))) - 1, 43_300_000)

    def test_accelerator_and_dma_address_map_is_stable(self) -> None:
        self.assertIn("0x43C00000", self.script)
        self.assertIn("0x40400000", self.script)

    def test_three_clock_domains_are_explicit(self) -> None:
        expected_frequencies = {
            0: "100.000000",
            1: "125.000000",
            2: "200.000000",
        }
        for clock_index, frequency in expected_frequencies.items():
            self.assertIn(
                f"CONFIG.PCW_FPGA{clock_index}_PERIPHERAL_FREQMHZ "
                f"{{{frequency}}}",
                self.script,
            )

        self.assertIn("[get_bd_pins ps7/FCLK_CLK0]", self.script)
        self.assertIn("[get_bd_pins ps7/FCLK_CLK1]", self.script)
        self.assertIn("[get_bd_pins ps7/FCLK_CLK2]", self.script)
        self.assertIn("[get_bd_pins accelerator/aclk]", self.script)
        self.assertIn("[get_bd_pins dma/m_axi_mm2s_aclk]", self.script)

    def test_control_and_stream_clock_crossings_are_explicit(self) -> None:
        for instance_name in ("accelerator_control_cdc", "dma_control_cdc"):
            self.assertRegex(
                self.script,
                rf"set {instance_name} .*axi_clock_converter:2\.1",
            )
        for instance_name in ("mm2s_cdc", "s2mm_cdc"):
            self.assertRegex(
                self.script,
                rf"set {instance_name} .*axis_clock_converter:1\.1",
            )

        self.assertIn("CONFIG.IS_ACLK_ASYNC {1}", self.script)
        self.assertIn("CONFIG.SYNCHRONIZATION_STAGES {3}", self.script)
        self.assertIn("mm2s_cdc/S_AXIS", self.script)
        self.assertIn("mm2s_cdc/M_AXIS", self.script)
        self.assertIn("s2mm_cdc/S_AXIS", self.script)
        self.assertIn("s2mm_cdc/M_AXIS", self.script)

    def test_descriptor_rtl_is_packaged(self) -> None:
        self.assertIn('"neuraedge_descriptor_sequencer.sv"', self.script)
        self.assertIn('"neuraedge_gradient_accumulator_bank.sv"', self.script)
        self.assertIn('"neuraedge_linear_backward8.sv"', self.script)
        self.assertIn("set_property range 256", self.script)

    def test_deployment_artifacts_require_physical_acceptance(self) -> None:
        self.assertIn(
            "STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE Explore",
            self.script,
        )
        self.assertIn("route_error_count != 0", self.script)
        self.assertIn("drc_error_count != 0", self.script)
        self.assertIn("setup_wns < 0.0", self.script)
        self.assertIn("hold_whs < 0.0", self.script)
        self.assertLess(
            self.script.index("if {$hold_whs < 0.0}"),
            self.script.index("file copy -force $generated_bit"),
            "the accepted bitstream must be published only after timing gates",
        )


if __name__ == "__main__":
    unittest.main()
