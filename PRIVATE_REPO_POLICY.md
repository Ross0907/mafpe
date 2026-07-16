# Private Repository Policy

This repository is the private, reproducible NeuraEdge Universal release.

- Include synthesizable RTL, software/compiler/runtime code, tests, scripts,
  board files, compact model artifacts, final routed hardware handoff files,
  native Vivado reports, representative SAIF, VCD/WDB waveforms, plots, and
  compiled documentation.
- Exclude virtual environments, downloaded research checkouts, raw CIFAR-10
  shard payloads, training checkpoints, Vivado project caches, XSim work
  directories, compiled simulator binaries, and duplicate timing iterations.
- Keep `Ross0907/mafpe` private before pushing any hardware or model artifact.
- Recompute `RELEASE_MANIFEST.json` and `RELEASE_MANIFEST.csv` with
  `generate_release_manifest.ps1` after changing release contents.
- Run `neuraedge_universal\scripts\run_regression.ps1` after RTL, protocol,
  compiler, or test source changes. Regenerate the release manifest after the
  regression updates its logs, then run `verify_private_release.ps1`.
- Label Vivado power output as a routed implementation estimate and retain its
  confidence/net-match data. Physical board measurements require a connected
  board and instrument metadata.

The canonical implementation target is PYNQ-Z2 `xc7z020clg400-1` with a
200 MHz accelerator clock, 125 MHz DMA/HP0 domain, and 100 MHz control domain.
