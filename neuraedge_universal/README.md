# NeuraEdge Universal

NeuraEdge Universal is a command-driven INT8, binary, ternary, and LIF tensor
accelerator targeting the PYNQ-Z2 (`xc7z020clg400-1`). The RTL combines a
64-DSP packed-MAC array, sparse reduction compaction, transformer vector
operators, persistent neuron state, an SGD update engine, and a descriptor
sequencer behind AXI DMA and AXI-Lite control.

## Reproduced model result

The source 4-timestep Spikingformer checkpoint measures **95.55%** on all
10,000 official CIFAR-10 test images. The exported integer-coefficient model
measures **95.42%**, a -0.13 percentage-point change. The 9,385,704-byte
parameter archive contains 118 aligned, hashed sections and is validated before
execution.

The runtime in `software/spikingformer_accelerator.py` maps the complete
tokenizer, LIF, pooling, four attention/MLP blocks, residual paths, reduction,
and classifier to accelerator commands. `software/run_spikingformer_pynq.py`
is the board entry point.

## Verify

From PowerShell in this directory:

```powershell
.\scripts\run_regression.ps1
```

This runs 93 Python unit/equivalence checks, strict Verilator lint, and all 15
RTL testbenches. The driver now requires each testbench's explicit `PASSED`
marker in addition to a zero simulator exit code. To reproduce the
largest-layer compiler benchmark:

```powershell
..\external_research\spikingformer_py310\Scripts\python.exe `
  .\software\benchmark_compiler_memory.py
```

## Build the PYNQ design

Vivado 2026.1 is discovered under `D:\AMDDesignTools\2026.1` by the scripts.
Validate the block-design contract without implementation:

```powershell
.\scripts\run_pynq_full.ps1 -Mode validate
```

Run synthesis, placement, route, timing, utilization, power, and bitstream
generation:

```powershell
.\scripts\run_pynq_full.ps1 -Mode full -CleanLogs
```

The final three-domain implementation uses 100 MHz AXI-Lite control, 125 MHz
DMA/HP0 memory transport, and a 200 MHz accelerator clock. Vivado 2026.1 routed
the complete PYNQ-Z2 design with setup WNS `+0.005 ns`, hold WHS `+0.050 ns`,
and zero unrouted nets. It uses 45,245 LUTs, 51,077 flip-flops, 110 DSP48E1s,
33 RAMB36s, and 8 RAMB18s.

Final board artifacts and native Vivado reports are under `outputs/pynq_full`:

- `neuraedge_pynq_full.bit`, `.hwh`, `.xsa`, and `_routed.dcp`
- `neuraedge_pynq_block_design.pdf` and `.svg`
- full timing, utilization, DRC, CDC, methodology, route, and power reports

The SAIF-annotated full-system estimate is 1.937 W total and 1.785 W dynamic,
with 7% of full-design nets matched and medium Vivado confidence. Vivado
attributes 0.472 W dynamic power to the accelerator hierarchy. This is a
routed implementation estimate from captured XSim activity, not a physical
power-rail measurement. The vectorless comparison is 2.152 W total.

## Run one CIFAR-10 sample on PYNQ

```powershell
python .\software\run_spikingformer_pynq.py `
  --bitfile .\outputs\pynq_full\neuraedge_pynq_full.bit `
  --manifest .\outputs\spikingformer_hardware_parameters\spikingformer_neuraedge_weights.json `
  --input sample.npy `
  --output result.json
```

The input can be raw `uint8` CIFAR layout or normalized floating-point CHW
layout. The JSON output records logits, prediction, elapsed time, throughput,
and the accelerator's hardware telemetry counters.

## Run full-backbone training on PYNQ

Deploy the overlay, parameter archive, runtime, and all hashed official
CIFAR-10 shards to a reachable board:

```powershell
.\scripts\deploy_to_pynq.ps1 -BoardHost xilinx@192.168.2.99
```

Start the default 400-epoch run, inspect it, request a checkpoint-boundary
pause, and resume the same optimizer state with:

```powershell
.\scripts\manage_pynq_training.ps1 -Action start -Epochs 400
.\scripts\manage_pynq_training.ps1 -Action status
.\scripts\manage_pynq_training.ps1 -Action pause
.\scripts\manage_pynq_training.ps1 -Action resume -Epochs 400
```

The board runtime executes the quantized forward graph, Q15 classifier loss
gradient, all affine/dynamic-attention backward products, four-step surrogate
BPTT, pool/residual reverse paths, and every parameter update through the PL
command ABI. Master parameters, momentum, and saved activations live in PYNQ
DDR; arithmetic and updates run in programmable logic. Checkpoints are written
atomically with the next epoch/minibatch coordinate, graph hash, Q formats, and
optimizer step, so resume starts at the first uncommitted minibatch.

`outputs/training/spikingformer_training_coverage.json` derives the complete
schedule from the exported graph: 43/43 execution operators, 30 learned affine
groups, 9,295,978 deployed trainable elements, and zero uncovered operators.

## Documentation

- `docs/architecture_contract.md`: measured architecture and verification rules
- `docs/stream_protocol.md`: commands, descriptor records, and register map
- `docs/neuraedge_full_architecture_explainer.pdf`: beginner-to-RTL architecture
- `docs/neuraedge_vivado_waveform_guide.pdf`: exact build/waveform workflow
- `outputs/deep_debug_final/neuraedge_deep_debug_report.pdf`: 55-page report
  containing 49 plots, all 15 waveform captures, and the native Vivado diagram
- `outputs/source_reference_final/neuraedge_line_by_line_source_reference.pdf`:
  3,277 pages covering 99 files and all 26,565 nonblank source lines
- `TOOLS_INSTALLED_AND_REMOVAL.txt`: tool locations and removal instructions

Vivado measurements are retained under `outputs/`; generated caches and
rebuildable intermediate files are excluded from source control.
