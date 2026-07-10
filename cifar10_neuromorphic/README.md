# SQIZH CIFAR-10 neuromorphic experiment

This folder runs the harder CIFAR-10 comparison path:

1. CIFAR-10 `3 x 32 x 32` images.
2. SQIZH spike-token transformer with latency/rate/Poisson input encodings.
3. Dense transformer baseline using the same patch size, embedding width, depth, and heads.
4. Compute-energy proxy using MAC/AC accounting and measured spike rates.
5. Plots and a PDF report in `outputs`.

Run:

```powershell
.\cifar10_neuromorphic\run_cifar10_pipeline.ps1
```

Quick verification:

```powershell
.\cifar10_neuromorphic\run_cifar10_pipeline.ps1 -Quick
```

The energy report is a compute proxy, not measured board power. It is intended to compare dense transformer work against an activity-gated hardware target where silent spikes skip patch/token operations.

## Paper-stack Spikingformer run

The local experimental trainer plateaued around 85% top-1. For the research-backed CIFAR-10 target, use the upstream Spikingformer trainer under the dedicated Python 3.10 environment:

```powershell
cd .\cifar10_neuromorphic
.\run_upstream_spikingformer_full.ps1 -Fresh
.\status_upstream_spikingformer.ps1
```

This path keeps the upstream Spikingformer model, timm data pipeline, mixup, optimizer, cosine schedule, AMP, and reset logic intact. The wrapper only adds Windows-safe background launch, checkpoint resume, pause files, logs, and JSON status.

Useful controls:

```powershell
.\pause_upstream_spikingformer.ps1
.\resume_upstream_spikingformer.ps1
.\status_upstream_spikingformer.ps1
..\external_research\spikingformer_py310\Scripts\python.exe .\generate_upstream_spikingformer_report.py
```

Default output:

```text
cifar10_neuromorphic\outputs_upstream_spikingformer\spikingformer_4_384_400e
```

The wrapper passes `--cooldown-epochs 0`, so `-Epochs 400` means exactly 400 scheduled epochs in status.

## Trained Spikingformer-to-RTL mapped accelerator

Use this trained, measured accelerator flow:

```powershell
.\cifar10_neuromorphic\run_spikingformer_head_accelerator.ps1
```

That command exports the trained `model_best.pth.tar` Spikingformer checkpoint, quantizes the 384-feature classifier head to fixed-point RTL memories, applies the measured low-magnitude feature activity gate, runs the full 10,000-image CIFAR-10 RTL fixture in Vivado XSim, validates every RTL class score against the fixed-point software reference, verifies the margin-gated on-chip learning update path against an independent Python golden model, and routes the AXI-stream PYNQ wrapper for the PYNQ-Z2/Zynq-7020 part.

Latest verified result:

```text
checkpoint:  outputs_upstream_spikingformer\spikingformer_4_384_400e\model_best.pth.tar
inference:   9551 / 10000 = 95.51%
score check: 0 RTL score mismatches, 0 prediction mismatches
feature gate: abs(feature) <= 2048 maps to zero before DSP multiply/update
activity:    57.81% measured feature-MAC skip opportunity on the exported CIFAR-10 test set
learning:    10000-sample RTL on-chip-learning run, 25 weight/bias updates
learn check: 0 score mismatches, 0 prediction mismatches, 0 update-flag mismatches, 0 update-count mismatches
rtl counters: inference skips 22,199,950 feature-MAC opportunities; learning skips 5,603 features
skip behavior: gated inference features bypass MAC_MUL/MAC_ACCUM; gated learning features bypass no-op true/pred weight writes
cycle saving: 44,399,900 inference cycles removed over the full 10,000-image fixture
top:         cifar10_spikingformer_head_pynq_axi_stream_top
part:        xc7z020clg400-1
clock:       10 ns / 100 MHz, timing met, routed WNS +0.336 ns, WHS +0.100 ns
resources:   1043 LUTs, 988 FFs, 228 LUTRAM LUTs, 6 RAMB36E1, 1 DSP48E1
power:       0.123 W total, 0.018 W dynamic, 0.105 W static vectorless
saif power:  0.116 W total, 0.011 W dynamic, 0.105 W static for both inference and learning wrapper traces, Vivado medium confidence
```

Outputs:

```text
cifar10_neuromorphic\outputs_spikingformer_head_accelerator
cifar10_neuromorphic\outputs_spikingformer_head_accelerator\vivado
cifar10_neuromorphic\outputs_spikingformer_head_accelerator\deep_debug
cifar10_neuromorphic\cifar10_spikingformer_head_results.csv
cifar10_neuromorphic\cifar10_spikingformer_head_onchip_learning_results.csv
cifar10_neuromorphic\cifar10_spikingformer_head_accelerator.vcd
cifar10_neuromorphic\cifar10_spikingformer_head_onchip_learning.vcd
```

Deep documentation and debug plots:

```powershell
.\external_research\spikingformer_py310\Scripts\python.exe .\cifar10_neuromorphic\generate_spikingformer_debug_reports.py
```

Activity-annotated power:

```powershell
.\cifar10_neuromorphic\run_spikingformer_head_activity_power.ps1
```

This compiles the AXI-stream wrapper activity testbench, generates full 10,000-sample inference and learning SAIF files from XSim, annotates the routed PYNQ checkpoint with `read_saif`, and writes `outputs_spikingformer_head_accelerator\activity_power`.

Generated reports:

```text
cifar10_neuromorphic\outputs_spikingformer_head_accelerator\deep_debug\neuraedge_spikingformer_deep_debug_report.pdf
cifar10_neuromorphic\outputs_spikingformer_head_accelerator\deep_debug\neuraedge_full_accelerator_line_by_line_explainer.pdf
cifar10_neuromorphic\outputs_spikingformer_head_accelerator\deep_debug\plots\*.png
```
