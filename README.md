# NeuraEdge Universal PYNQ-Z2 Accelerator

This private repository contains the complete command-driven INT8, binary,
ternary, spiking-transformer, backward, and SGDW accelerator release for
PYNQ-Z2 (`xc7z020clg400-1`). It includes RTL, the Python compiler/runtime,
full-backbone training schedules, tests, Vivado hardware handoff artifacts,
waveforms, SAIF activity, debug plots, and line-by-line documentation.

## Verified model and graph

- Source 4-timestep Spikingformer CIFAR-10 accuracy: **95.55%**
  (`9,555 / 10,000` official test images).
- Exported integer-coefficient accuracy: **95.42%**
  (`9,542 / 10,000`), a `-0.13` percentage-point change.
- Float/integer prediction disagreements: `255 / 10,000`.
- Hardware parameter archive: `9,385,704` bytes, 118 aligned sections,
  SHA-256 `b02c5df519eaecffb228f4c8be21babddc01761f26210bb8dfcca3318ed0d381`.
- Complete traced graph test: label `3`, prediction `3`, maximum logit error
  `9.596289141100556e-08`, using 1,345,393 commands.

The mapped graph covers the tokenizer, four timesteps, LIF state, pooling,
four spiking-attention/MLP blocks, residual paths, reduction, and classifier.

## Full-backbone on-device learning

The training compiler covers all `43 / 43` execution operators, 30 affine
groups, and 9,295,978 deployed trainable elements. It emits 14,742,724
backward partial tiles, 7,474,847,252 signed backward multiplies, and
1,161,998 optimizer commands. Forward, backward, surrogate-gradient, and SGDW
arithmetic execute in programmable logic; board DDR stores master parameters,
momentum, saved activations, and atomic resumable checkpoints.

## Final PYNQ implementation

Vivado 2026.1 generated the complete processing-system, AXI DMA, clock/reset
CDC, interconnect, and accelerator overlay:

| Result | Final value |
|---|---:|
| Accelerator/control/DMA clocks | 200 / 100 / 125 MHz |
| Setup WNS / TNS | +0.005 ns / 0.000 ns |
| Hold WHS / THS | +0.050 ns / 0.000 ns |
| Routed nets / routing errors | 74,557 / 0 |
| DRC errors | 0 |
| LUT / FF | 45,245 / 51,077 |
| DSP48E1 / RAMB36 / RAMB18 | 110 / 33 / 8 |

Artifact SHA-256 values:

- BIT: `d9bcac6224efd44da8aa2f8a0e674f5e66a67294caac460d6cfc8086d5a341de`
- HWH: `79395d99357fa798ecd31b0c9229500dce7e7aada8185c3576b744fd4485712f`
- XSA: `8d06b8a0b6e82105992cbf05fd1f0d3cfac8f5316d682f976f88221b0583bd64`
- routed DCP: `750e7185560d3187c7576f84b722918d3c8d3da21738d417b8fa59fc9ca857a1`

## Power evidence

Vivado's routed SAIF-annotated estimate is **1.937 W total**, **1.785 W
dynamic**, with **0.472 W** attributed to the accelerator hierarchy. The report
has medium confidence and matches `9,164 / 123,250` implementation nets; the
remainder is modeled probabilistically. The vectorless comparison is 2.152 W
total and 1.993 W dynamic. These are implementation estimates with the exact
SAIF windows and reports retained in the repository.

## Reproduce

Run the complete independent regression:

```powershell
cd .\neuraedge_universal
.\scripts\run_regression.ps1
```

Build and sign off the full overlay:

```powershell
.\scripts\run_pynq_full.ps1 -Mode full -CleanLogs
```

Capture all 15 waveform sets and rerun routed SAIF power:

```powershell
.\scripts\capture_waveforms.ps1 -OutputDirectory .\outputs\waveforms_final
.\scripts\run_saif_power.ps1 `
  -Checkpoint .\outputs\pynq_full\neuraedge_pynq_full_routed.dcp `
  -OutputDirectory .\outputs\power_activity_full `
  -SaifDirectory .\outputs\power_activity_full `
  -AcceleratorInstance system_i/accelerator/inst
```

Verify this private package and all committed hashes:

```powershell
cd ..
.\verify_private_release.ps1
```

## Documentation and evidence

- `docs/neuraedge_full_architecture_explainer.pdf`: beginner-to-RTL system
  explanation.
- `docs/neuraedge_vivado_waveform_guide.pdf`: exact Vivado build, schematic,
  waveform, timing, and SAIF workflow.
- `docs/neuraedge_line_by_line_source_reference.pdf`: 3,277 pages covering 99
  files, 28,915 physical lines, and every one of 26,565 nonblank lines.
- `neuraedge_universal/outputs/deep_debug_final/neuraedge_deep_debug_report.pdf`:
  55 pages and 49 plots, including all 15 waveform views.
- `neuraedge_universal/outputs/pynq_full`: BIT/HWH/XSA/DCP and native signoff
  reports.

Raw CIFAR-10 shards, virtual environments, research checkouts, Vivado project
caches, and duplicate timing iterations are intentionally excluded. Their
creation/removal details are in `TOOLS_INSTALLED_AND_REMOVAL.txt` and the
project scripts.
