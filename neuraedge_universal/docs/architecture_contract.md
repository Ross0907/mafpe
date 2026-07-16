# NeuraEdge Universal Accelerator Architecture Contract

## 1. Scope and target

The implementation target is the PYNQ-Z2 device `xc7z020clg400-1`. The design
must accelerate the arithmetic used by conventional CNNs, standard and hybrid
vision transformers, and directly trained spiking transformers. It must not
depend on a fixed CIFAR-10 classifier shape.

The accelerator clock target is 200 MHz. The routed board design deliberately
uses separate 100 MHz AXI-Lite control and 125 MHz AXI DMA/HP0 domains. AMD AXI
clock-converter IP crosses control and 64-bit AXI-Stream traffic into the 200
MHz accelerator domain with asynchronous FIFOs and three synchronizer stages.
Any reported frequency must come from a routed Vivado timing report for this
exact part. The design is allowed to use off-chip DDR for complete model weights
because the Zynq-7020 has 140 RAMB36 tiles, far less storage than a
multi-million-parameter model. Tiles and partial sums remain on chip during a
command.

## 2. Measured model and arithmetic baseline

The source Spikingformer checkpoint reaches 95.55 percent (9,555/10,000) on
the official CIFAR-10 test set when evaluated in floating point. The exported
integer coefficient model reaches 95.42 percent (9,542/10,000); 255 predictions
differ from the floating-point checkpoint, a measured accuracy change of
-0.13 percentage points. These are full-test-set software measurements, not a
small sample or a classifier-only measurement.

The hardware archive contains 118 aligned and individually hashed sections:
29 folded convolution/linear projections plus the final classifier. Its exact
size is 9,385,704 bytes and its SHA-256 digest is
`b02c5df519eaecffb228f4c8be21babddc01761f26210bb8dfcca3318ed0d381`.
The executable runtime maps every tokenizer, LIF, max-pool, attention, residual,
MLP, temporal-reduction, and classifier stage to accelerator commands. Archive
loading rejects a changed whole-file hash, a changed section hash, overlapping
sections, inconsistent shapes, and out-of-range offsets before DMA begins.

## 3. Research-derived design decisions

1. One INT8 GEMM engine serves convolution and fully connected layers. Host or
   streaming im2col changes convolution into GEMM. This follows the released
   FlexViT PYNQ-Z2 architecture and avoids separate underutilized engines.
2. Each command chooses input-broadcast or weight-broadcast traversal using a
   byte-traffic estimate. This addresses the shape difference between wide
   projection layers and tall, narrow convolution layers.
3. A complete K tile is accumulated in local registers before output. This is
   the depth-first, output-stationary rule used to avoid 32-bit partial-sum DDR
   traffic in FlexViT and is consistent with the IO-awareness principle of
   FlashAttention.
4. Standard transformer nonlinear operations use integer arithmetic. The
   vector path includes integer square-root LayerNorm and power-of-two
   attention probabilities based on I-ViT and FQ-ViT.
5. Spiking paths are not emulated with dense multiplies. Binary activations use
   conditional addition, ternary activations use conditional add/subtract, and
   alpha-XNOR similarity gives matching spikes weight 1 and matching zeros a
   programmable `alpha` in `[0,1]`, exactly as defined by Xiao et al.
6. Multi-timestep data can be batched and stored in banks so independent time
   steps are processed without repeatedly loading membrane state. The reason
   is the temporal-parallel architecture reported by Chen and Chang.
7. All-zero reduction positions are compacted when activations load. Compute
   iterates over the compact index list, so sparsity removes cycles rather than
   merely multiplying by zero.
8. A descriptor sequencer accepts complete tensor programs rather than making
   software issue and poll every tile command. Each record declares its input
   and output lengths and carries a final-record flag. The sequencer validates
   record boundaries and output counts in hardware, while the host partitions
   DMA packets only between records when the AXI DMA length limit is reached.

Primary sources:

- FlexViT: https://arxiv.org/abs/2606.31938
- FlashAttention: https://arxiv.org/abs/2205.14135
- I-ViT: https://arxiv.org/abs/2207.01405
- FQ-ViT: https://arxiv.org/abs/2111.13824
- Alpha-XNOR SSA: https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html
- A2OS2A: https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.pdf
- Temporal-parallel spiking accelerator: https://arxiv.org/abs/2503.19643

## 4. Dense tensor engine

The physical array contains 64 packed DSP MAC blocks. Each block computes two
exact unsigned 8-bit products, giving 128 MAC lanes per cycle. At 200 MHz this
is a peak of 25.6 GMAC/s or 51.2 GOPS when multiplication and addition are counted
as separate operations.

The tile is eight rows by 16 output columns. Two arrangements share the same
DSPs:

- Outer mode: eight independent rows reuse 16 weights for one K position.
- Dot8 mode: eight adjacent K values are reduced for 16 output columns. This
  keeps narrow matrix-vector and classifier operations from using only one of
  the eight row lanes.

The K capacity is 2,048 elements. This covers the 1,536-wide Spikingformer MLP
and the largest reduction dimension identified in the FlexViT model survey.

## 5. Exact asymmetric-INT8 arithmetic

The packed DSP consumes bytes `A` and `B`. For activation zero point `Za`,
weight zero point `Zb`, reduction length `K`, row sum `Ra`, and column sum `Cb`,
the result is:

```text
raw       = sum(A * B)
corrected = raw - Zb*Ra - Za*Cb + K*Za*Zb + bias
```

This is algebraically identical to `sum((A-Za)*(B-Zb)) + bias`. The independent
Python test compares these two expressions on 100 randomized matrix shapes and
zero points.

The DSP packing uses:

```text
packed_weight = weight_lo + (weight_hi << 17)
packed_product = packed_weight * activation
product_lo = packed_product[15:0]
product_hi = packed_product[32:17]
```

Bit 16 is a guard. Since `255*255 = 65025 < 65536`, the low product cannot
carry into the upper field. Because bit 24 makes the packed operand look signed
to a DSP48E1, RTL adds `activation << 25` through the DSP C port whenever that
bit is set. Vivado maps the complete corrected expression to one `C + A*B`
DSP48E1.

## 6. Neuromorphic modes

- Binary weighted mode computes `sum(spike ? weight : 0)` with no multiplier.
- Ternary weighted mode computes `sum(+weight, 0, or -weight)` for values in
  `{-1,0,1}` and supports A2OS2A-style ternary value paths.
- Alpha-XNOR mode accumulates `count(1,1)*256 + count(0,0)*alpha_q8`.
- LIF mode applies leak, current integration, threshold comparison, reset, and
  spike generation with persistent membrane state.

## 7. Full transformer operator coverage

The command set must cover:

- GEMM and im2col convolution
- bias, residual add, and requantization
- ReLU, binary spike, ternary spike, and LIF
- integer LayerNorm
- dense QK dot-product attention
- power-of-two log-integer softmax and AV aggregation
- alpha-XNOR spiking attention
- pooling and classifier projection

Model software tiles larger tensors, selects reuse mode, emits commands and
quantization metadata, and checks unsupported dimensions before execution.

## 8. Executable Spikingformer dataflow

`SpikingformerAcceleratorModel` is the end-to-end hardware execution graph for
the exported 4-timestep, 384-channel checkpoint. It implements the checkpoint's
actual forward path rather than substituting a generic transformer template:

1. A normalized `3 x 32 x 32` image is quantized and sent through five folded
   convolution projections. LIF stages run over all four timesteps and two
   binary max-pool stages reduce the spatial size to `4 x 4`.
2. The first tokenizer projection is evaluated once and broadcast in time,
   matching the source model's temporal-input construction exactly.
3. Each of four transformer blocks builds 12-head Q, K, and V tensors. Binary
   QK similarity runs through the conditional-add tensor path; unsigned AV
   aggregation uses the dense path. The checkpoint's `0.125` attention scale,
   `0.5` attention threshold, output projection, and residual are preserved.
4. Each block's two-layer spiking MLP executes `384 -> 1536 -> 384` and adds its
   residual. The source checkpoint does not invoke LayerNorm in this forward
   graph, so the runtime does not insert one merely because conventional
   transformers normally contain it.
5. Global temporal/spatial averaging produces 384 features. The signed INT8
   classifier produces ten 32-bit accumulators and dequantized logits.

The manifest totals 3,732,164,352 model MAC equivalents, 104,256 tile commands,
243,330,192 raw tile-input bytes, and 52,297,808 output values. These counts are
generated from the exported tensor shapes and compiler tiling rules.

## 9. PYNQ execution and DMA boundaries

The Zynq processing system communicates with the accelerator over AXI DMA and
AXI-Lite control. Scatter-gather length width is set to 26 bits, making the
legal packet limit 67,108,863 bytes instead of the IP default of 16,383 bytes.
The driver still checks this limit and partitions a descriptor program only at
record boundaries; accelerator state is retained between packets.

The measured compiler benchmark for the largest block-3 projection shape,
`1024 x 1728 x 384`, produces 6,168 instructions, 43,130,880 payload bytes,
148,032 descriptor bytes, 43,278,912 total input bytes, and 3,145,728 output
bytes. The vectorized compiler completes this case in 0.334 seconds in the
recorded run, compared with 95.601 seconds for the former scalar packer.

The physical board graph uses three clock/reset islands:

- FCLK0 at 100 MHz drives PS GP0, AXI-Lite interconnect, and software-visible
  register traffic.
- FCLK1 at 125 MHz drives AXI DMA, the 64-bit HP0 DDR path, and memory
  interconnect. Its theoretical stream width-rate product is 1.0 GB/s before
  protocol and DDR effects.
- FCLK2 at 200 MHz drives the accelerator RTL. Dedicated AXI-Lite and
  AXI-Stream clock converters isolate every crossing; no raw multibit bus is
  sampled directly across domains.

## 10. Verification contract

No implementation claim is accepted unless all applicable gates pass:

1. Independent Python algebra/unit tests.
2. Vivado XSim directed, randomized, backpressure, reset, and boundary tests.
3. Icarus four-state simulation for portable modules.
4. Verilator strict lint with warnings reviewed.
5. Bit-exact RTL CSV comparison to the independent Python model.
6. Full model accuracy on the model's standard validation set when the dataset
   is locally available; otherwise operator/model-output equivalence is labeled
   separately and is not called an accuracy result.
7. Vivado synthesis, placement, route, DRC, timing, utilization, and DCP.
8. XSim-generated SAIF annotation for estimated dynamic power.
9. Board measurements are reported only when collected from a physical board.

## 11. Final routed PYNQ-Z2 evidence

Vivado 2026.1 completed synthesis, placement, physical optimization, routing,
bitstream generation, and hardware handoff for the complete processing-system,
DMA, clock-converter, interconnect, and accelerator design. The routed result is:

- setup WNS `+0.005 ns`, setup TNS `0.000 ns`, zero failing setup endpoints;
- hold WHS `+0.050 ns`, hold THS `0.000 ns`, zero failing hold endpoints;
- 74,557 fully routed nets and zero routing errors;
- 45,245 used LUTs, 51,077 flip-flops, 110 DSP48E1 blocks, 33 RAMB36 blocks,
  and 8 RAMB18 blocks;
- zero DRC errors. The 122 DRC warnings are named DSP pipelining, LUT equation,
  RAM asynchronous-control, and one no-routable-load advisory; all are retained
  in the detailed report rather than suppressed;
- three methodology warnings are LUT-driven asynchronous-reset checks inside
  AMD AXI interconnect FIFO logic, not the accelerator RTL;
- 26 synchronized single-bit CDC structures and 146 CDC-15 warnings located
  inside the two AMD XPM asynchronous stream FIFOs, each protected by the IP's
  max-delay/false-path constraints.

The vectorless routed estimate is 2.152 W total and 1.993 W dynamic. Relocated
XSim SAIF activity matches 9,164 of 123,250 full-design nets and produces a
1.937 W total, 1.785 W dynamic estimate, including 0.472 W in the accelerator
hierarchy. Both reports have medium Vivado confidence. Untouched simulated
storage produces documented X-state warnings, and neither estimate is called a
physical PYNQ-Z2 power measurement.

The final artifact SHA-256 values are:

- BIT: `d9bcac6224efd44da8aa2f8a0e674f5e66a67294caac460d6cfc8086d5a341de`
- HWH: `79395d99357fa798ecd31b0c9229500dce7e7aada8185c3576b744fd4485712f`
- XSA: `8d06b8a0b6e82105992cbf05fd1f0d3cfac8f5316d682f976f88221b0583bd64`
- routed DCP: `750e7185560d3187c7576f84b722918d3c8d3da21738d417b8fa59fc9ca857a1`

## 12. Packed-DSP feasibility gate

`neuraedge_mac2_u8` passed 131,072 exhaustive RTL vectors. A 128-pair probe
routed on `xc7z020clg400-1` at 5.0 ns with exactly 128 DSP48E1 cells, 1,609 LUTs,
4,117 registers, setup WNS `+1.250 ns`, hold WHS `+0.042 ns`, and no synthesis
warnings or errors. This result fixes the production array width; later blocks
must still pass their own full implementation checks.
