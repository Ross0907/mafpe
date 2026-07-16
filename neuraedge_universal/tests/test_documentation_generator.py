from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generate_source_reference import (
    generate,
    project_root,
    python_statement_spans,
    source_entries,
    sv_contexts,
    sv_statement_spans,
)


class DocumentationGeneratorTests(unittest.TestCase):
    def test_every_nonblank_source_line_is_covered_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tex_path, manifest_path = generate(Path(temporary_directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries = source_entries(project_root())

            self.assertEqual(manifest["summary"]["file_count"], len(entries))
            self.assertEqual(
                manifest["summary"]["nonblank_lines"],
                manifest["summary"]["explained_nonblank_lines"],
            )
            self.assertTrue(tex_path.read_text(encoding="utf-8").endswith("\\end{document}\n"))
            generated_tex = tex_path.read_text(encoding="utf-8")
            self.assertIn("Beginner architecture map", generated_tex)
            self.assertIn("What becomes FPGA hardware", generated_tex)
            self.assertIn("Two unsigned byte weights", generated_tex)
            listing_paths = [
                line.removeprefix("\\lstinputlisting{").removesuffix("}")
                for line in generated_tex.splitlines()
                if line.startswith("\\lstinputlisting{")
            ]
            self.assertEqual(len(listing_paths), len(entries))
            for listing_path in listing_paths:
                self.assertTrue(
                    (tex_path.parent / listing_path).resolve().is_file(),
                    listing_path,
                )
            for file_record in manifest["files"]:
                self.assertEqual(
                    file_record["nonblank_lines"],
                    file_record["explained_nonblank_lines"],
                )
                self.assertEqual(len(file_record["sha256"]), 64)

    def test_rtl_context_returns_to_structure_after_clocked_process(self) -> None:
        lines = [
            "always_ff @(posedge clk) begin",
            "  state <= next_state;",
            "end",
            "logic later_signal;",
        ]
        contexts = sv_contexts(lines)
        self.assertIn("clocked process", contexts[2])
        self.assertEqual(contexts[4], "the module's structural declarations")

    def test_multiline_statements_are_explained_as_complete_operations(self) -> None:
        rtl_lines = [
            "always_ff @(posedge clk) begin",
            "  result_r <= left_r",
            "      + right_r;",
            "end",
        ]
        rtl_spans = sv_statement_spans(rtl_lines)
        self.assertEqual((rtl_spans[2].start, rtl_spans[2].end), (2, 3))
        self.assertIn("left_r + right_r", rtl_spans[3].text)

        python_text = "value = function(\n    operand_a,\n    operand_b,\n)\n"
        python_spans = python_statement_spans(python_text)
        self.assertEqual((python_spans[1].start, python_spans[1].end), (1, 4))
        self.assertIn("operand_a", python_spans[3].text)


if __name__ == "__main__":
    unittest.main()
