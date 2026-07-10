# NeuraEdge CIFAR-10 PYNQ Accelerator Private Release

This private repository contains the clean accelerator release package for the CIFAR-10 Spikingformer-head / PYNQ-Z2 hardware flow.

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

A full zipped artifact bundle is stored under `artifacts/`. Browsable source, scripts, manifests, and reports are committed directly as normal repository files.
