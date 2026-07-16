from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from generate_debug_report import (
    choose_wave_signals,
    parse_cdc_summary,
    parse_clock_timing,
    parse_drc_summary,
    parse_implementation_summary,
    parse_power_report,
    parse_utilization_hierarchy,
    parse_vcd,
)


class DebugReportTests(unittest.TestCase):
    def test_vcd_parser_keeps_exact_fsm_state_ahead_of_clear_state(self) -> None:
        fixture = """$timescale 1ps $end
$scope module top $end
$var reg 1 ! clk $end
$var reg 1 \" clear_state $end
$scope module dut $end
$var reg 3 # state [2:0] $end
$var reg 1 $ program_active $end
$upscope $end
$upscope $end
$enddefinitions $end
#0
0!
0\"
b0 #
0$
#5
1!
b1 #
1$
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fixture.vcd"
            path.write_text(fixture, encoding="utf-8")
            timescale, signals = parse_vcd(path)

        selected = choose_wave_signals(signals)
        leaves = [signal.name.rsplit("/", 1)[-1] for signal in selected]
        self.assertEqual(timescale, "1ps")
        self.assertIn("state", leaves)
        self.assertIn("program_active", leaves)
        self.assertNotIn("clear_state", leaves)
        self.assertEqual(signals["/top/dut/state"].transitions[-1], (5, 1.0))

    def test_final_vivado_report_parsers_recover_verified_metrics(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reports = root / "outputs" / "pynq_full"
        implementation = parse_implementation_summary(
            reports / "full_implementation_summary.txt"
        )
        clocks = parse_clock_timing(reports / "full_routed_timing_summary.rpt")
        utilization = parse_utilization_hierarchy(
            reports / "full_routed_utilization_hierarchical.rpt"
        )
        drc = parse_drc_summary(reports / "full_drc.rpt")
        cdc = parse_cdc_summary(reports / "full_cdc.rpt")
        power = parse_power_report(
            root / "outputs" / "power_activity_full" / "vivado_reports"
            / "routed_saif_power.rpt"
        )

        self.assertEqual(
            clocks["clk_fpga_2"],
            (implementation["setup_wns_ns"], implementation["hold_whs_ns"]),
        )
        self.assertEqual(
            utilization["system_wrapper"]["dsp"],
            int(implementation["dsp48_count"]),
        )
        self.assertFalse(any(row[1] == "Error" for row in drc))
        self.assertGreater(sum(row[3] for row in drc), 0)
        self.assertTrue(any(row[0] == "CDC-15" for row in cdc))
        self.assertGreater(power["total"], 0.0)
        self.assertRegex(power["matched"], r"\d+%\s+\(\d+/\d+\)")


if __name__ == "__main__":
    unittest.main()
