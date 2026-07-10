# NeuraEdge CIFAR-10 PYNQ Accelerator Private Release

This private repository is the clean release bundle for the CIFAR-10 Spikingformer-head neuromorphic accelerator, including RTL, fixed-point memories, XSim fixtures, Vivado routed reports, SAIF activity-power reports, debug plots, and line-by-line documentation.

Canonical result, verified July 10, 2026:

- CIFAR-10 RTL accuracy: 9,551 / 10,000 = 95.51%.
- RTL score mismatches vs independent fixed-point Python reference: 0.
- RTL prediction mismatches vs independent fixed-point Python reference: 0.
- On-chip classifier-learning updates: 25.
- On-chip learning mismatches vs independent Python online-update model: 0.
- Feature gate: `abs(feature) <= 2048`.
- RTL inference skip counter: 22,199,950 skipped feature-MAC opportunities.
- RTL learning skip counter: 5,603 skipped learning features.
- Implemented skip behavior: inference skips bypass `ST_MAC_MUL` and `ST_MAC_ACCUM`; learning skips bypass no-op true/pred weight updates.
- Inference cycles removed by the skip FSM: 44,399,900 over the full 10,000-image fixture.
- Vivado target: PYNQ-Z2/Zynq-7020, `xc7z020clg400-1`, 10 ns / 100 MHz.
- Routed timing: WNS +0.336 ns, WHS +0.100 ns, timing constraints met.
- Routed resources: 1,043 LUTs, 988 FFs, 228 LUTRAM LUTs, 6 RAMB36E1, 1 DSP48E1.
- Vectorless Vivado power: 0.123 W total, 0.018 W dynamic, 0.105 W static.
- XSim-SAIF Vivado power: 0.116 W total, 0.011 W dynamic, 0.105 W static for both inference and learning wrapper traces, Vivado medium confidence.
- SAIF direct net match: 262 / 3079 nets for inference and 294 / 3079 nets for learning; Vivado probabilistically propagates unmatched activity.

Main entry points:

```powershell
.\setup_neuraedge_python.ps1
.\run_neuraedge_private_full.ps1
.\cifar10_neuromorphic\run_spikingformer_head_accelerator.ps1 -SkipExport -Samples 10000 -Part xc7z020clg400-1 -ClockPeriodNs 10.0
.\cifar10_neuromorphic\run_spikingformer_head_activity_power.ps1 -SkipInferenceSaif -SkipLearningSaif
```

The package deliberately excludes local virtual environments, datasets, checkpoints, and transient Vivado/XSim project caches. See `RELEASE_MANIFEST.json` for every included file and SHA-256 hash.