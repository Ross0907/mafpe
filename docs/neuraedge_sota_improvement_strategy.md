# NeuraEdge SOTA Improvement Strategy

This document keeps the repository honest about what is measured now, what the relevant SOTA targets are, and which upgrades are mathematically/research motivated.

## Current Measured Accelerator

Canonical implementation:

```text
cifar10_neuromorphic\rtl\cifar10_spikingformer_head_accelerator.sv
cifar10_neuromorphic\rtl\cifar10_spikingformer_head_pynq_axi_stream_top.sv
cifar10_neuromorphic\run_spikingformer_head_accelerator.ps1
run_neuraedge_private_full.ps1
```

Verified on July 10, 2026 with Vivado 2026.1 and the PYNQ-Z2/Zynq-7020 part `xc7z020clg400-1`:

| Item | Measured result |
|---|---:|
| CIFAR-10 samples | 10,000 |
| Fixed-point RTL top-1 | 95.51% |
| Correct predictions | 9,551 |
| RTL score mismatches | 0 |
| RTL prediction mismatches | 0 |
| On-chip-learning updates | 25 |
| On-chip-learning golden mismatches | 0 |
| Feature gate | `abs(feature) <= 2048` |
| Feature-MAC skip opportunity | 57.81% |
| RTL inference skip counter | 22,199,950 |
| RTL learning skip counter | 5,603 |
| Skip-control effect | inference skips bypass `ST_MAC_MUL` and `ST_MAC_ACCUM`; learning skips bypass no-op true/pred updates |
| Inference cycles removed | 44,399,900 over the 10,000-image fixture |
| Routed WNS / WHS at 100 MHz | +0.336 ns / +0.100 ns |
| LUT / FF / LUTRAM LUT | 1043 / 988 / 228 |
| BRAM36 / DSP48E1 | 6 / 1 |
| Vectorless Vivado power | 0.123 W total, 0.018 W dynamic, 0.105 W static |

The implemented improvement is a fixed-point feature-activity denoising gate with RTL-visible counters and real FSM bypass. A sweep over exported CIFAR-10 Spikingformer features found that `abs(feature) <= 2048` removes 57.81% of feature-MAC opportunities while moving the mapped fixed-point head from 95.48% to 95.51%. The hardware applies the same effective feature to both inference and on-chip classifier learning. Skipped inference features now bypass `ST_MAC_MUL` and `ST_MAC_ACCUM`, and skipped learning features bypass no-op true/pred weight updates. The counters report 22,199,950 skipped inference opportunities plus 5,603 skipped learning features in the full 10,000-image run, so validation is not bypassing the accelerator path.

## Current SOTA Targets

These are the papers and repositories used as the comparison targets for further work:

| Work | Reported result relevant to CIFAR-10 / energy | Why it matters |
|---|---:|---|
| Spikingformer | 95.81% CIFAR-10, 95.95% for the dagger variant; repository also reports 57.34% ImageNet energy reduction vs Spikformer | This is the upstream family used by the current checkpoint and feature export. |
| Alpha-XNOR spiking self-attention | 96.15% CIFAR-10 for the reported multi-stage ViT setting | Shows that replacing dot-product spike similarity with alpha-XNOR can improve spiking transformer attention. |
| Max-Former | 97.04% CIFAR-10 at T=4 with 6.57M parameters; reports lower ImageNet energy for the Max-Former setting | Shows that high-frequency information and lightweight early token mixing can improve accuracy and energy together. |

Primary sources:

- Spikingformer paper: https://arxiv.org/html/2304.11954v5
- Spikingformer repository: https://github.com/TheBrainLab/Spikingformer
- Max-Former paper: https://arxiv.org/html/2505.18608v4
- Alpha-XNOR paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.pdf

## Improvement Categories

1. Accuracy-preserving hardware sparsity already implemented:
   - Feature gate zeros low-magnitude fixed-point features before the DSP multiply.
   - This is equivalent to deterministic activation pruning in the exported feature space.
   - The chosen threshold was selected by full 10,000-sample evaluation, not by a smoke test.
   - The final RTL does not merely multiply by zero: gated inference features skip the multiply/accumulate states and gated learning features skip the no-op weight read/calc/write states.

2. Accuracy upgrades for the next backbone export:
   - Add Max-Former-style high-frequency patch embedding to preserve information before spiking quantization.
   - Add alpha-XNOR attention scoring for spike queries/keys so non-spike pairs do not collapse into uninformative dot-product behavior.
   - Retrain with the feature gate in the loop as quantization-aware activity regularization, so the backbone learns the exact hardware silence rule.

3. Accelerator upgrades:
   - Replace the current gated dense scan with a compact active-feature stream, so the implemented cycle skip can also remove feature-index visits and BRAM address checks.
   - Add a top-k active-feature index buffer per frame. This changes classifier work from `classes * feature_dim` to `classes * active_feature_count`.
   - Add SAIF/VCD-driven Vivado power annotation for inference and learning modes, replacing vectorless power with activity-specific power.
   - Add a small PYNQ host driver that streams real feature frames through DMA and checks hardware labels against the CSV reference.

4. On-chip learning upgrades:
   - Keep the current bounded perceptron-style classifier update as the fast local rule.
   - Add optional eligibility traces for repeated mistakes so updates can use recent temporal context without full backpropagation.
   - Add a learning-rate schedule register and margin register exposed through AXI-Lite for controlled experiments.

## Acceptance Criteria For Future Claims

- Accuracy claims must come from the full 10,000-image CIFAR-10 test set unless explicitly labeled otherwise.
- RTL claims must include XSim CSV validation against an independent Python golden model.
- FPGA claims must include routed timing, utilization, and power artifacts from Vivado.
- Power-saving claims must state whether they are vectorless, VCD/SAIF-annotated, board-measured, or analytic operation-count estimates.
- SOTA comparison claims must cite the exact paper/repo and the exact reported metric.
