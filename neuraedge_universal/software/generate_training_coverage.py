"""Generate the audited full-graph NeuraEdge training coverage manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuraedge_training import build_spikingformer_training_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-manifest", required=True, type=Path)
    parser.add_argument("--parameter-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = build_spikingformer_training_coverage(
        arguments.execution_manifest,
        arguments.parameter_manifest,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    summary = result["summary"]
    print(
        "NEURAEDGE_TRAINING_COVERAGE_PASSED "
        f"operators={summary['covered_execution_operator_count']} "
        f"learned_groups={summary['learned_affine_group_count']} "
        f"parameters={summary['trainable_deployed_parameter_elements']} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
