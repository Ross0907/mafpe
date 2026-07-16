# NeuraEdge Command and AXI Stream Protocol

This document is the byte-accurate software/RTL contract implemented by
`neuraedge_universal_core.sv`, `neuraedge_buffered_tensor_core.sv`,
`neuraedge_descriptor_sequencer.sv`, and `neuraedge_pynq_axi_top.sv`. The
executable host implementation is
`software/neuraedge_protocol.py`, and `tests/test_neuraedge_protocol.py` checks
the bit positions, signed encodings, beat ordering, and `TLAST` placement.

## 1. Transfer rules

- A command is 128 bits. Software writes bits `[31:0]`, `[63:32]`, `[95:64]`,
  and `[127:96]` to AXI-Lite registers `0x10`, `0x14`, `0x18`, and `0x1c`, then
  writes control bit 0 at address `0x00` to submit it.
- Stream data is 64 bits per beat. A beat transfers only when both `TVALID` and
  `TREADY` are high on the same rising clock edge.
- Multi-beat input commands require `TLAST=1` only on the final specified beat.
  Early or missing `TLAST` sets the sticky error bit.
- Output must be consumed until the beat carrying `TLAST=1`. Backpressure is
  legal on every output beat; payload and `TLAST` remain stable while stalled.
- Signed fields use two's-complement representation. Byte and bit vectors are
  little-lane ordered: lane zero occupies the least-significant field.

## 2. AXI-Lite registers

| Offset | Access | Meaning |
|---:|:---:|---|
| `0x00` | R/W | bit 0 submit one command; bit 1 clear done/wrapper error; bit 2 start descriptor program; bit 8 interrupt enable |
| `0x04` | R | bit 0 busy, bit 1 sticky error, bit 2 done, bit 3 command pending, bit 4 interrupt, bit 5 descriptor program active |
| `0x08` | R | active opcode in bits `[3:0]` |
| `0x0c` | R | total submitted command count |
| `0x10..0x1c` | R/W | four command words, least-significant first |
| `0x20` | R | accepted input stream beat count |
| `0x24` | R | accepted output stream beat count |
| `0x28` | R | target clock in hertz, `200000000` |
| `0x2c` | R | capability bits, currently `0xff`; bit 7 denotes descriptor programs |
| `0x30` | R | ABI version, currently `0x00010007` |
| `0x34` | R | ASCII identity `NEUR`, `0x4e455552` |
| `0x38` | R | loaded activation issues `[11:0]`; loaded weight issues `[27:16]` |
| `0x3c` | R | compacted active tensor issues `[11:0]` |
| `0x40` | R | tensor compute cycles for the latest tensor command |
| `0x44` | R | skipped all-zero tensor issues |
| `0x48/0x4c` | R | actual dense MAC count, low/high 32-bit words |
| `0x50/0x54` | R | actual binary/ternary additions, low/high 32-bit words |
| `0x58` | R | LayerNorm cycle count |
| `0x5c` | R | LayerNorm signed mean |
| `0x60` | R | LayerNorm variance |
| `0x64` | R | LayerNorm integer sigma |
| `0x68` | R | log-softmax cycle count |
| `0x6c` | R | sign-extended log-softmax maximum score |
| `0x70` | R | log-softmax denominator `[23:0]` |
| `0x74/0x78` | R | cumulative LIF spikes, low/high 32-bit words |
| `0x7c` | R | completed eight-lane optimizer updates |
| `0x80` | R | loaded activation-LUT entries `[8:0]` |
| `0x84` | R | LayerNorm reciprocal in unsigned Q31 |
| `0x88` | R | log-softmax reciprocal in unsigned Q31 |
| `0x8c` | R | descriptor records completed by the current or most recent program |
| `0x90` | R/W | total output beats expected from the next program |

The dense-MAC, sparse-add, stream-beat, LIF-spike, and optimizer-update
counters accumulate across commands. Opcode `0xf` clears the child-engine
counters; control bit 1 clears wrapper-side done/error state. Software reads
64-bit counters low word first and then high word after the accelerator is
idle, so the two words describe one stable value.

The PYNQ overlay sets AXI DMA's buffer-length width to 26 bits. One direct-mode
MM2S or S2MM packet may therefore contain at most `2^26-1 = 67,108,863` bytes,
rather than the IP default of 14 bits. The largest command payload is only
32 KiB, and the largest current full-model layer program is approximately
43.3 MB input and 6.3 MB output, so each fits in one legal DMA packet.

### Descriptor programs

A descriptor program submits many ordinary 128-bit commands and all their
input payloads in one MM2S DMA transfer. This removes per-tile AXI-Lite and DMA
setup overhead without creating a second command encoding: every descriptor
contains the same command bits accepted by the single-command interface.

Each record is a sequence of 64-bit stream beats:

| Beat | Bits | Meaning |
|---:|---:|---|
| 0 | `[31:0]` | number of payload beats following this command |
| 0 | `[62:32]` | number of output beats this command must produce |
| 0 | `[63]` | final-record flag; set only on the final record |
| 1 | `[63:0]` | command bits `[63:0]` |
| 2 | `[63:0]` | command bits `[127:64]` |
| `3..` | `[63:0]` | exactly the payload count declared in beat 0 |

Before starting, software writes the sum of all record output counts to `0x90`,
arms the S2MM DMA when output is nonempty, and writes control bit 2. The final
flag in beat 0 identifies the last record; `0x8c` counts records actually
completed so software can verify the submitted sequence afterward. The input
DMA asserts its external `TLAST` only on the last beat of the whole program.
The sequencer regenerates a command-local
`TLAST` on each record's final payload beat and checks every declared input and
output count. On output, external `TLAST` is asserted only on the final beat of
the whole program, including when the program ends with one or more no-output
commands. A malformed header, record count, final flag, command completion, or
stream boundary sets sticky error and drains the remaining input through its
external `TLAST` so the DMA channel can recover.

## 3. Opcodes

| Opcode | Name | Input | Output |
|---:|---|---:|---:|
| `1` | load activations | `K` beats | none |
| `2` | load outer weights | `2K` beats | none |
| `3` | load Dot8 weights | `16K8` beats | none |
| `4` | dense outer compute | none | `rows*columns` beats |
| `5` | dense Dot8 compute | none | `columns` beats |
| `6` | binary weighted compute | none | `rows*columns` beats |
| `7` | ternary weighted compute | none | `rows*columns` beats |
| `8` | Alpha-XNOR | 8 beats | 1 beat |
| `9` | vector ALU | 9 beats | 4 beats |
| `a` | integer LayerNorm | `2N` beats | `N` beats |
| `b` | log-integer softmax | `N` beats | `N` beats |
| `c` | 32-lane LIF step | 8 beats | 1 beat |
| `d` | 8-lane SGDW update | 8 beats | 8 beats |
| `e` | activation LUT | depends on suboperation | depends on suboperation |
| `f` | clear sticky status/counters | none | none |

Here `K` is an outer-product issue count in `[1,2048]`, `K8` is a group of
eight reduction positions in `[1,256]`, and `N` is the operator item count.

## 4. Tensor command

Tensor opcodes use this common command layout:

| Bits | Width | Field |
|---:|---:|---|
| `[3:0]` | 4 | opcode |
| `[7:4]` | 4 | active output rows, compute commands require `1..8` |
| `[13:8]` | 6 | active output columns, compute commands require `1..16` |
| `[25:14]` | 12 | issue count |
| `[33:26]` | 8 | unsigned activation zero point `Za` |
| `[41:34]` | 8 | unsigned weight zero point `Zb` |
| `[127:42]` | 86 | reserved, write zero |

### Activation load

Each beat contains eight activation bytes, lanes 0 through 7 in bits `[7:0]`
through `[63:56]`. One beat represents one K position in outer mode or eight
adjacent K positions for the single-row Dot8 mode. The hardware compares all
eight bytes with `Za`; all-zero-point issues are omitted from the active-index
list and therefore consume no tensor-array issue cycle.

### Outer weight load

Each K position has 16 weight bytes and therefore uses two beats. Beat 0 holds
columns 0 through 7 and beat 1 holds columns 8 through 15. Maximum storage is
`2048*16 = 32768` bytes.

### Dot8 weight load

Each issue is an `8 x 16` block. The 16 beats are ordered as reduction lane 0
columns 0..7, reduction lane 0 columns 8..15, then the same two halves for
reduction lanes 1 through 7. Maximum storage is again
`256*8*16 = 32768` bytes.

### Tensor output

| Bits | Meaning |
|---:|---|
| `[47:0]` | signed 48-bit accumulated result |
| `[51:48]` | output row index |
| `[57:52]` | output column index |
| `[58]` | final-result flag, identical to stream `TLAST` |
| `[63:59]` | zero |

Dense results implement exactly
`sum((A-Za)*(B-Zb))`. The RTL obtains this from the raw unsigned products using
`raw - Zb*row_sum - Za*column_sum + K*Za*Zb`. Binary mode conditionally adds a
weight for activation value 1. Ternary mode adds for `+1`, subtracts for `-1`,
and skips zero.

## 5. Alpha-XNOR

The command stores active bit count in `[50:42]` and unsigned Q8 alpha in
`[59:51]`; each is in `[1,256]` except alpha may be zero. Input beats 0..3 are
the 256 query bits and beats 4..7 are the 256 key bits. Within every word, the
lowest-numbered bit occupies the least-significant position.

Output bits `[17:0]` are the Q8 score, `[26:18]` are one/one matches, and
`[35:27]` are zero/zero matches. The score is
`256*one_matches + alpha_q8*zero_matches`.

## 6. Vector ALU

Command fields are operation `[7:4]`, signed Q15 multiplier `[23:8]`, right
shift `[28:24]`, signed output zero point `[44:29]`, signed threshold `[76:45]`,
and unsigned surrogate window `[108:77]`. Operations 0..8 are saturating add,
ReLU, requantize, binary spike, ternary spike, surrogate-gradient gate, maximum,
signed Q15 multiply, and clamp. Operation 9 is flexible requantize. For that
operation only, `[28:24]` is a signed binary-point shift from -15 through 15
and the result scale is `multiplier_q15 / 2^(15 + shift)`. This covers folded
scales above one while preserving the complete legacy operation-2 encoding.
The unused signed value -16 is rejected as a configuration error.

Input beat 0 contains signed clamp minimum `[31:0]` and maximum `[63:32]`.
Beats 1..4 hold the eight signed 32-bit A lanes, two lanes per beat. Beats 5..8
hold B in the same format. Output is four beats containing eight signed 32-bit
lanes. `TLAST` is asserted on output beat 3.

## 7. LayerNorm

The command stores feature count `1..2048` in `[15:4]` and epsilon in `[47:16]`.
Each feature uses two input beats. Beat 0 holds signed 24-bit value `[23:0]`,
signed 24-bit gamma `[47:24]`, and beta bits `[15:0]` in `[63:48]`. Beat 1
bits `[7:0]` hold beta bits `[23:16]`; its other bits are zero. Only the second
beat of the final feature carries `TLAST`.

Each output beat contains signed result `[31:0]`, feature index `[43:32]`, and
final flag `[44]`. The implementation uses integer mean, integer variance,
non-restoring square root, one shared Q31 reciprocal, Q15 gamma, and integer
beta. It does not call a floating-point IP.

## 8. Log-integer softmax

The command stores item count `1..256` in `[12:4]` and difference shift in
`[17:13]`. Every input beat contains one signed 24-bit score in `[23:0]`.
Every output beat contains unsigned Q15 probability `[15:0]`, item index
`[24:16]`, and final flag `[25]`.

The engine subtracts each score from the maximum, right-shifts the difference,
uses `2^(15-exponent)` as the unnormalized probability, sums the denominator,
and multiplies by one Q31 reciprocal. This is a power-of-two integer
approximation, not IEEE exponential softmax.

## 9. LIF step

The command stores leak shift `[8:4]`, signed 24-bit threshold `[32:9]`, signed
24-bit reset `[56:33]`, and clear-state flag `[57]`. Eight input beats hold 32
signed 16-bit currents, four lanes per beat. The output low 32 bits are spike
bits and high 32 bits are the persistent timestep count. Membrane state stays
in fabric across commands unless clear-state is set.

## 10. SGDW optimizer

The command stores unsigned Q16 learning rate `[21:4]`, momentum `[38:22]`, and
weight decay `[55:39]`. Learning rate is limited to 131071; momentum and decay
are limited to 65536. Each of eight input beats contains signed weight `[15:0]`,
signed gradient `[39:16]`, and signed velocity `[63:40]`. Each output beat
returns updated signed weight `[15:0]` and velocity `[39:16]`; `[63:40]` is zero.

The lane update is `v'=sat24(round(v*momentum/2^16)+g)`. The separately
decayed update direction is
`d=sat24(v'+round(w*decay/2^16))`, followed by
`w'=sat16(w-round(d*learning_rate/2^update_right_shift))`. Weight decay does
not enter the retained velocity, which is the defining SGDW decoupling from
the momentum state. All three intermediate round/saturate boundaries match
the RTL and bit-exact software reference.

## 11. Activation LUT

Suboperation `[7:4]` is 0 clear, 1 load, or 2 evaluate. Item count is `[19:8]`.
Load accepts `1..256` beats, each with address `[7:0]` and signed 16-bit value
`[23:8]`. Evaluate accepts one beat containing eight byte indices and returns
two beats containing eight signed 16-bit table values. Clear has no stream data.

## 12. Error recovery

An invalid field, wrong count, wrong weight format, early/late `TLAST`, or child
operator configuration error sets the sticky error bit. Software must drain any
already-produced output, wait for busy to clear, issue opcode `0xf`, write
control bit 1 to clear any wrapper-side error, and verify that status error
cleared before submitting more work. Reset also clears all state, including LIF
membrane and the activation LUT.

## 13. Compiler mapping

`software/neuraedge_compiler.py` converts a matrix operation into this public
ABI; it does not access internal RTL state. For dense asymmetric uint8 GEMM it
implements these rules:

1. Split output rows into groups of eight and columns into groups of 16.
2. Split K only when it exceeds the 2,048-element local operand capacity.
3. Use outer mode for two or more output rows. Use Dot8 for one output row,
   padding the final K group with the exact activation and weight zero points.
4. Choose input-broadcast or weight-broadcast traversal from the total operand
   bytes each traversal reloads. A caller can force either mode for experiments.
5. Pad inactive row/column lanes with their zero points, but request only the
   true row and column count in the compute command.
6. Decode and validate every returned row/column tag and `TLAST`, accumulate
   independent K chunks in signed int64, then add bias exactly once.

`compile_spike_gemm` follows the same tiling but emits binary or ternary compute
commands. It writes signed INT8 weights as two's-complement bytes; binary spikes
are `0/1`, and ternary spikes are `0x00/0x01/0xff`. Its reported
`estimated_spike_adds` is `nonzero_spikes * output_columns`, the number of useful
conditional additions rather than the equivalent dense MAC count.

The test-side `TensorTransactionEmulator` independently decodes the packed beat
sequence and emulates persistent operand memories. Compiler tests cover edge
tiles, both traversal orders, Dot8 padding, asymmetric zero points, reductions
that cross the local-memory boundary, binary spikes, ternary spikes, signed
weight boundaries, result-tag reconstruction, and bias placement.
