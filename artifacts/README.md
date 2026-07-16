# Hardware Artifacts

The canonical PYNQ-Z2 handoff files are stored without duplication under:

`neuraedge_universal/outputs/pynq_full`

That directory contains:

- `neuraedge_pynq_full.bit`
- `neuraedge_pynq_full.hwh`
- `neuraedge_pynq_full.xsa`
- `neuraedge_pynq_full_routed.dcp`
- `artifact_sha256.txt`
- native timing, utilization, route, DRC, CDC, methodology, congestion, power,
  clock-interaction, and block-design exports

Run `..\verify_private_release.ps1` from this directory, or
`.\verify_private_release.ps1` from the repository root, to validate the
release manifest and the four canonical hardware hashes.
