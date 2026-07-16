"""Generate an auditable, context-aware line-by-line source reference.

The generated TeX contains the complete source text and one explanation for
every nonblank line.  Explanations use the enclosing Python function or RTL
process, known accelerator signal meanings, and the exact left/right sides of
assignments.  A JSON manifest records every included file and SHA-256 digest so
the PDF can be checked against the source tree that produced it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FILE_PURPOSES = {
    "rtl/neuraedge_activation_buffer.sv": "stores one local tile of unsigned eight-bit activations for repeated tensor-array reads",
    "rtl/neuraedge_activation_lut8.sv": "loads and evaluates a programmable 256-entry activation lookup table on eight lanes",
    "rtl/neuraedge_active_index_buffer.sv": "stores the reduction indices that remain after all-zero activation positions are compacted",
    "rtl/neuraedge_alpha_xnor.sv": "computes alpha-weighted XNOR spiking similarity and streams its accumulated scores",
    "rtl/neuraedge_buffered_tensor_core.sv": "loads activation and weight tiles, compacts sparse reduction positions, schedules the physical tensor tile, and isolates result backpressure with a registered skid FIFO",
    "rtl/neuraedge_command_pkg.sv": "defines the shared command opcodes and tensor-mode encodings used by RTL and host software",
    "rtl/neuraedge_descriptor_sequencer.sv": "turns a multi-record DMA program into checked core commands and payload/output streams",
    "rtl/neuraedge_gradient_accumulator_bank.sv": "stores one lane of exact cross-command backward gradients in inferred distributed memory",
    "rtl/neuraedge_integer_layernorm.sv": "evaluates integer LayerNorm by accumulating moments, taking an integer square root, dividing, scaling, and saturating",
    "rtl/neuraedge_isqrt64.sv": "computes an exact floor square root of an unsigned 64-bit radicand with an iterative restoring algorithm",
    "rtl/neuraedge_lif_bank.sv": "stores 32 persistent membrane values and applies leak, integration, threshold, reset, and spike counting",
    "rtl/neuraedge_linear_backward8.sv": "computes exact tiled linear-layer input and weight gradients with a reusable four-by-four signed DSP array",
    "rtl/neuraedge_log_softmax.sv": "implements power-of-two integer softmax normalization without floating-point exponent hardware",
    "rtl/neuraedge_mac2_array_probe.sv": "instantiates many packed MAC blocks solely to measure their routed FPGA resource and timing behavior",
    "rtl/neuraedge_mac2_u8.sv": "packs two exact unsigned eight-bit multiplications into one DSP48E1 multiplication",
    "rtl/neuraedge_pynq_axi_top.sv": "provides AXI-Lite control, AXI-Stream DMA, interrupts, descriptor mode, and hardware telemetry for PYNQ",
    "rtl/neuraedge_sgd_optimizer8.sv": "applies signed fixed-point SGD with momentum to eight weight lanes and stores updated state",
    "rtl/neuraedge_tensor_tile.sv": "implements the eight-row by sixteen-column dense, binary, and ternary arithmetic array",
    "rtl/neuraedge_udivider.sv": "performs deterministic unsigned integer division one quotient bit per cycle",
    "rtl/neuraedge_universal_core.sv": "decodes commands and arbitrates the tensor, vector, normalization, LIF, LUT, attention, backward, and optimizer engines through a staged result path",
    "rtl/neuraedge_vector_alu.sv": "executes lane-wise residual, activation, comparison, requantization, and pooling primitives",
    "rtl/neuraedge_weight_bank.sv": "stores one 16-column by 2,048-entry weight tile in banked FPGA memory",
    "software/benchmark_compiler_memory.py": "measures time, memory, command count, and byte traffic for the largest real Spikingformer projection",
    "software/evaluate_spikingformer.py": "evaluates the source checkpoint and exported integer arithmetic over the complete CIFAR-10 test set",
    "software/export_cifar10_shards.py": "writes deterministic CIFAR-10 training and validation shards for bounded-memory board-side learning",
    "software/export_spikingformer_hardware.py": "folds BatchNorm, calibrates quantizers, and writes aligned hashed hardware parameter sections",
    "software/generate_debug_report.py": "turns full-test predictions, compiler manifests, XSim waveforms, and final Vivado reports into auditable plots and PDFs",
    "software/generate_operator_fixtures.py": "builds independent LayerNorm, softmax, vector, optimizer, and LUT vectors for RTL tests",
    "software/generate_source_reference.py": "generates this source-reference volume and its source-integrity manifest",
    "software/generate_spikingformer_manifest.py": "derives every model tensor shape, tile schedule, command count, operation count, and traffic total",
    "software/generate_training_coverage.py": "records which Spikingformer parameters receive hardware gradients and optimizer updates in each training policy",
    "software/generate_tensor_tile_fixture.py": "builds deterministic dense, binary, and ternary tensor vectors for bit-exact RTL comparison",
    "software/neuraedge_compiler.py": "tiles matrices and convolutions into validated accelerator commands and packed stream payloads",
    "software/neuraedge_golden.py": "implements independent mathematical reference models for every hardware arithmetic primitive",
    "software/neuraedge_protocol.py": "defines command bit fields, descriptor records, packed streams, and result decoders",
    "software/neuraedge_pynq.py": "controls the overlay, AXI-Lite registers, DMA allocation, packet partitioning, errors, and telemetry",
    "software/neuraedge_training.py": "builds quantized forward, backward, surrogate-gradient, and optimizer programs for hardware-resident supervised updates",
    "software/run_spikingformer_pynq.py": "loads one CIFAR-10 image and executes the complete exported Spikingformer graph on a PYNQ overlay",
    "software/spikingformer_accelerator.py": "maps all tokenizer, spiking attention, MLP, residual, reduction, and classifier stages to hardware operations",
    "software/spikingformer_parameters.py": "validates and exposes typed read-only views of the aligned parameter archive",
    "software/spikingformer_quantization.py": "defines calibration, signed/unsigned affine quantization, rounding, saturation, and folded projection math",
    "software/spikingformer_reference.py": "reconstructs the trained source graph and its integer hardware-emulation path for accuracy comparison",
    "software/spikingformer_training_runtime.py": "runs resumable CIFAR-10 training through the accelerator command ABI with atomic checkpoints and validation",
    "software/train_spikingformer_pynq.py": "provides the board-training command line, validates artifacts, and coordinates resume, pause, epochs, and reports",
}


FILE_DATAFLOWS = {
    "rtl/neuraedge_activation_buffer.sv": "A load command writes activation bytes into addressed entries. During compute, the scheduler presents reduction indices and this memory returns one activation vector per selected row; keeping the tile local avoids rereading DDR for every output column.",
    "rtl/neuraedge_activation_lut8.sv": "Software first streams address/value pairs into the 256-entry table. A later execute command reads eight addresses in parallel, returns eight programmed nonlinear values, and counts only newly initialized addresses so duplicate writes cannot falsely report a complete table.",
    "rtl/neuraedge_active_index_buffer.sv": "While activations load, the tensor core tests whether a reduction position is all zero across active rows. Only nonzero positions are written here; compute later walks this compact list, so sparse positions remove issue cycles rather than performing useless zero arithmetic.",
    "rtl/neuraedge_alpha_xnor.sv": "The input stream supplies packed query and key spike masks. The block counts matching one bits and matching zero bits, weights zero matches by programmable alpha, accumulates across chunks, and emits one fixed-point attention score under valid/ready backpressure.",
    "rtl/neuraedge_buffered_tensor_core.sv": "This is the tile-level controller around memories and the physical array. Commands are predecoded, validated, and then routed through activation load, weight load, sparse-index formation, issue FIFO, accumulation, and ordered result output states. A two-entry skid FIFO reserves one beat for registered READY latency, decouples the tile's 128-entry result store from parent backpressure, and delays command completion until the buffered final result is actually transferred.",
    "rtl/neuraedge_descriptor_sequencer.sv": "A DMA record header declares one command's payload and result lengths. The sequencer captures both halves of the 128-bit command, forwards exactly the declared payload, counts accepted results, regenerates final TLAST, and drains malformed packets so the next descriptor starts aligned.",
    "rtl/neuraedge_integer_layernorm.sv": "The block first stores a vector while accumulating its sum and squared sum. It derives integer mean and variance, invokes the iterative square-root and divider, applies per-element gamma and beta, saturates the result, and preserves output data whenever downstream ready is low.",
    "rtl/neuraedge_isqrt64.sv": "One iteration examines the next two radicand bits, shifts the partial remainder, conditionally subtracts the trial divisor, and appends one root bit. After 32 iterations the 32-bit result is exactly floor(sqrt(radicand)); a widened subtraction bit makes borrow detection explicit.",
    "rtl/neuraedge_lif_bank.sv": "Each of 32 lanes reads persistent membrane state, applies the leak shift, adds signed input current, compares against threshold, emits a spike, and writes either reset or integrated membrane back. Segmented counters avoid a long telemetry carry chain on the 200 MHz datapath.",
    "rtl/neuraedge_linear_backward8.sv": "A command declares up to eight rows, input features, and output features. Streamed activations, signed weights, and output gradients are buffered in four-lane halves; sixteen signed DSP products are accumulated per cycle. The controller first reduces dY times transposed W into dX, then transposed X times dY into dW, retaining 47-bit partial sums across tiled commands when requested. Separate magnitude, rounding-bias, shift, sign, and saturation states preserve the reference round-once result while breaking the long FPGA timing path.",
    "rtl/neuraedge_log_softmax.sv": "The engine buffers logits, finds their maximum, converts bounded negative differences to power-of-two terms, sums the denominator, divides each term into a fixed probability scale, and streams normalized outputs without a floating-point exponential unit.",
    "rtl/neuraedge_mac2_u8.sv": "Two unsigned byte weights are separated by a guard bit and multiplied by one activation in a DSP48E1. Registered correction handles the DSP's signed interpretation, then slices recover two independent exact 16-bit products without cross-field carry.",
    "rtl/neuraedge_pynq_axi_top.sv": "This is the software-visible shell. AXI-Lite writes configure command and descriptor registers, AXI-Stream carries DMA data, registered credit boundaries break READY timing paths, and status/interrupt registers expose completion, errors, and telemetry.",
    "rtl/neuraedge_sgd_optimizer8.sv": "For eight lanes, the engine reads weight, gradient, and momentum velocity, computes the fixed-point momentum update, rounds the learning-rate product symmetrically, saturates the new signed weight, stores new velocity/weight state, and streams the result.",
    "rtl/neuraedge_tensor_tile.sv": "The 8-by-16 tile schedules 64 packed DSP blocks as 128 dense byte MAC lanes, or bypasses multipliers for binary/ternary conditional additions. Per-row active masks, registered sparse popcount, issue/result FIFOs, and deterministic row/column output order make backpressure safe.",
    "rtl/neuraedge_udivider.sv": "A restoring divider shifts one dividend bit into the remainder each cycle, compares/subtracts the divisor, and writes one quotient bit. Divide-by-zero is reported explicitly instead of producing an accidental numeric value.",
    "rtl/neuraedge_universal_core.sv": "This command arbiter gives exactly one engine ownership of the shared streams. It stages LUT loads, LIF input, and other multibeat commands, then captures each engine into a source-local result bank before a registered shared output stage. Completion waits for the final AXI transfer, so stalls cannot acknowledge a command early; local accept signals prevent an inactive engine's counters from entering another engine's timing cone.",
    "rtl/neuraedge_vector_alu.sv": "Eight signed lanes execute residual add, ReLU, requantization, spike conversion, surrogate-gradient gating, maximum, Q15 multiply, or clamp. Wide intermediate products are registered before rounding and saturation so fixed-point behavior is deterministic.",
    "rtl/neuraedge_weight_bank.sv": "Each bank stores sixteen weight bytes for every reduction position. Eight banks feed outer mode rows or adjacent reduction groups, allowing the same memory organization to support matrix tiles and narrow dot-product schedules.",
    "software/neuraedge_compiler.py": "Validated NumPy tensors are tiled into hardware-sized activation and weight loads followed by compute records. The compiler chooses outer or dot8 reuse from traffic, lowers grouped convolution through im2col, packs little-endian stream beats, and rejects dimensions the RTL cannot represent.",
    "software/neuraedge_protocol.py": "This file is the bit-level contract shared with RTL. It packs each 128-bit command field, declares exact payload/result lengths, wraps records in descriptor headers, and decodes signed results so a software integer has the same bit meaning as FPGA wires.",
    "software/neuraedge_pynq.py": "The board driver loads the BIT/HWH overlay, allocates physically contiguous DMA buffers, writes AXI-Lite registers, arms receive before send, waits with timeouts, checks hardware errors, and returns telemetry beside result words.",
    "software/neuraedge_training.py": "The training compiler converts saved forward activations and class loss derivatives into descriptor programs. It emits surrogate-gradient gates for spike nonlinearities, exact tiled linear-backward commands for dX and dW, and Q16 optimizer commands for persistent weight and momentum state. Every payload length, gradient shape, numeric shift, and returned update is checked against the same protocol helpers used by RTL tests.",
    "software/spikingformer_accelerator.py": "The exported model graph calls compiler primitives in checkpoint order: tokenizer convolutions and LIF, four Q/K/V attention and MLP blocks, residuals, reduction, and classifier. Tensor shapes and quantization records, rather than a hard-coded class result, determine every hardware program.",
    "software/export_spikingformer_hardware.py": "Checkpoint weights and BatchNorm statistics are folded into affine projections, calibrated, quantized, aligned to 64-byte sections, and hashed. The JSON manifest records offsets, shapes, dtypes, scales, and section digests consumed by both emulator and board runtime.",
    "software/generate_debug_report.py": "This evidence builder reads immutable result artifacts instead of rerunning or inventing measurements. It parses VCD transitions and Vivado text tables, plots all standard-test predictions and hardware reports, embeds the native block design, and records every plot's source artifact.",
    "software/spikingformer_reference.py": "The source checkpoint is reconstructed with its original timestep and spike behavior. A separate hardware-emulation path applies exported integer coefficients, allowing full-test accuracy and per-logit disagreement to be measured before board execution.",
    "software/spikingformer_training_runtime.py": "Each epoch loads deterministic CIFAR-10 shards, executes the quantized forward graph, forms the supervised loss gradient, and dispatches hardware backward and optimizer descriptor programs. Updated parameter and velocity arrays are checkpointed atomically with epoch, sample cursor, RNG state, hashes, and metrics, so pause or power loss resumes from a committed minibatch boundary instead of replaying an ambiguous update.",
    "software/train_spikingformer_pynq.py": "The command-line entry point validates the overlay, parameter archive, dataset shards, and ABI before constructing the runtime. It translates pause-file and resume options into explicit checkpoint behavior, reports board telemetry and validation accuracy, and exits nonzero when required physical artifacts or numerical checks fail.",
}


IDENTIFIER_MEANINGS = {
    "clk": "the clock that makes all registers advance together",
    "aclk": "the AXI and accelerator clock",
    "rst_n": "the active-low synchronous reset",
    "aresetn": "the active-low AXI reset",
    "valid": "the producer's statement that its current payload is meaningful",
    "ready": "the consumer's statement that it can accept a payload this cycle",
    "last": "the AXI-Stream marker for the final beat of a packet or command payload",
    "state": "the finite-state-machine phase retained in flip-flops",
    "command": "the 128-bit instruction selecting an engine, shape, mode, and numeric parameters",
    "descriptor": "a DMA record header plus one command and its declared payload",
    "input_remaining": "the number of command payload beats still expected",
    "output_remaining": "the number of command result beats still expected",
    "program_output_remaining": "the number of result beats left in the complete descriptor program",
    "activation": "an input feature value consumed by a neuron or multiply-accumulate operation",
    "weight": "a learned coefficient multiplied by an activation",
    "bias": "an additive learned offset applied after accumulation",
    "accumulator": "a widened running sum that prevents intermediate overflow",
    "membrane": "the persistent LIF neuron state carrying charge between timesteps",
    "threshold": "the membrane level at which a neuron emits a spike",
    "leak": "the retained fraction of prior membrane state",
    "spike": "a binary event produced when a neuron crosses its threshold",
    "dense_mac": "a count of ordinary activation-times-weight operations actually issued",
    "spike_add": "a count of event-driven conditional weight additions actually issued",
    "zero_point": "the integer code representing real numeric zero in affine quantization",
    "scale": "the real-value spacing represented by one quantized integer step",
    "requant": "the conversion of a wide accumulated result back into a bounded activation code",
    "alpha": "the programmable weight assigned to matching zero spikes in alpha-XNOR attention",
    "gamma": "the learned multiplicative LayerNorm parameter",
    "beta": "the learned additive LayerNorm parameter",
    "mean": "the arithmetic average used by normalization",
    "variance": "the average squared distance from the mean",
    "gradient": "the signed loss derivative used to update a learned parameter",
    "velocity": "the momentum history retained by the SGD optimizer",
    "learning_rate": "the fixed-point multiplier controlling update magnitude",
    "dma": "the direct-memory-access engine moving contiguous buffers between DDR and FPGA logic",
    "interrupt": "the hardware notification that wakes software after completion or error",
    "counter": "a hardware telemetry register that accumulates measured events",
    "records": "the number of descriptor records accepted since program start",
    "payload": "stream data belonging to the current command",
    "buffer": "local storage that decouples producer and consumer timing",
    "index": "an integer selecting one tensor element, lane, bank, or state entry",
    "lane": "one parallel arithmetic element within a vector or tensor datapath",
    "row": "one output-row position processed by the tensor tile",
    "column": "one output-feature position processed in parallel",
    "timestep": "one discrete SNN update instant",
    "logits": "the ten unnormalized classifier scores used for argmax",
    "prediction": "the class index with the greatest classifier score",
}


STATE_MEANINGS = {
    "IDLE": "waits for a new operation and does not consume payload data",
    "HEADER": "accepts and validates the descriptor record header",
    "COMMAND_LOW": "captures command bits 63 through 0",
    "COMMAND_HIGH": "captures command bits 127 through 64 and checks input TLAST",
    "SUBMIT": "holds command-valid until the universal core accepts the instruction",
    "PAYLOAD": "forwards exactly the declared number of input beats to the selected engine",
    "WAIT": "waits for both command completion and the declared output count",
    "ERROR_DRAIN": "discards remaining DMA input until TLAST so the next transfer starts aligned",
    "LOAD": "stores incoming operands into local FPGA memory",
    "COMPUTE": "issues arithmetic work using values already held on chip",
    "OUTPUT": "presents computed results under valid/ready backpressure",
    "COUNT": "updates a segmented telemetry counter without a long carry chain",
    "DONE": "raises completion for one cycle and returns control to idle",
}


@dataclass(frozen=True)
class SourceEntry:
    path: Path
    relative: str
    purpose: str


@dataclass(frozen=True)
class StatementSpan:
    start: int
    end: int
    text: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_entries(root: Path) -> list[SourceEntry]:
    patterns = (
        ("rtl", "*.sv"),
        ("software", "*.py"),
        ("tb", "*.sv"),
        ("tests", "*.py"),
        ("scripts", "*.ps1"),
        ("scripts", "*.tcl"),
    )
    entries: list[SourceEntry] = []
    for directory, pattern in patterns:
        for path in sorted((root / directory).glob(pattern)):
            relative = path.relative_to(root).as_posix()
            purpose = FILE_PURPOSES.get(relative, inferred_file_purpose(relative))
            entries.append(SourceEntry(path, relative, purpose))
    return entries


def inferred_file_purpose(relative: str) -> str:
    stem = Path(relative).stem
    words = humanize(stem.removeprefix("tb_").removeprefix("test_"))
    if relative.startswith("tb/"):
        return f"drives, backpressures, and checks the {words} RTL block in simulation"
    if relative.startswith("tests/"):
        return f"checks {words} behavior independently with Python assertions"
    if relative.endswith(".tcl"):
        return f"runs the reproducible Vivado {words} hardware flow"
    if relative.endswith(".ps1"):
        return f"orchestrates the repeatable Windows {words} command-line flow"
    return f"implements the {words} part of the project"


def humanize(identifier: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
    return text.replace("_", " ").replace("-", " ").strip().lower()


def identifier_meaning(identifier: str) -> str:
    lowered = identifier.lower().removesuffix("_r").removesuffix("_i").removesuffix("_o")
    if lowered in IDENTIFIER_MEANINGS:
        return IDENTIFIER_MEANINGS[lowered]
    matches = [meaning for key, meaning in IDENTIFIER_MEANINGS.items() if key in lowered]
    if matches:
        return matches[0]
    if lowered.endswith("_mem") or "memory" in lowered:
        return "local FPGA storage addressed by surrounding index logic"
    if lowered.endswith("_count") or lowered.endswith("_counter"):
        return "an integer count retained for control or measurable telemetry"
    if lowered.endswith("_valid"):
        return "a handshake flag saying the associated payload is meaningful"
    if lowered.endswith("_ready"):
        return "a handshake flag saying the associated consumer can accept data"
    if lowered.endswith("_data"):
        return "the payload carried with the surrounding valid/ready handshake"
    if lowered.endswith("_enable") or lowered.endswith("_en"):
        return "a control bit that permits the associated state or storage update"
    return f"the value named '{humanize(identifier)}' in this block"


def summarize_expression(expression: str) -> str:
    identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_$]*\b", expression)
    ignored = {
        "if", "else", "begin", "end", "logic", "wire", "signed", "unsigned",
        "true", "false", "int", "return", "self", "np", "Path",
    }
    selected: list[str] = []
    for identifier in identifiers:
        if identifier.lower() in ignored or identifier in selected:
            continue
        selected.append(identifier)
        if len(selected) == 3:
            break
    if not selected:
        return "The expression is a literal or structural operator and has no hidden software state."
    meanings = "; ".join(f"{name}: {identifier_meaning(name)}" for name in selected)
    return f"The important operands are {meanings}."


def python_contexts(text: str) -> dict[int, str]:
    contexts: dict[int, tuple[int, str]] = {}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            description = f"the {kind} {node.name}, which handles {humanize(node.name)}"
            span = end - node.lineno
            for line_number in range(node.lineno, end + 1):
                previous = contexts.get(line_number)
                if previous is None or span < previous[0]:
                    contexts[line_number] = (span, description)
    return {line: value[1] for line, value in contexts.items()}


def _joined_statement(lines: list[str], start: int, end: int) -> str:
    return " ".join(
        fragment for fragment in (line.strip() for line in lines[start - 1:end])
        if fragment
    )


def python_statement_spans(text: str) -> dict[int, StatementSpan]:
    """Map multiline host statements to their complete source expression."""
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    candidates: list[ast.AST] = []
    simple_nodes = (
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise,
        ast.Assert, ast.Import, ast.ImportFrom,
    )
    for node in ast.walk(tree):
        if isinstance(node, simple_nodes):
            candidates.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            candidates.append(node)
    result: dict[int, StatementSpan] = {}
    for node in candidates:
        start = int(getattr(node, "lineno", 0))
        end = int(getattr(node, "end_lineno", start))
        if start <= 0 or end <= start:
            continue
        span = StatementSpan(start, end, _joined_statement(lines, start, end))
        for line_number in range(start, end + 1):
            previous = result.get(line_number)
            if previous is None or (end - start) < (previous.end - previous.start):
                result[line_number] = span
    return result


def sv_statement_spans(lines: list[str]) -> dict[int, StatementSpan]:
    """Recover multiline RTL assignments without pretending fragments are complete."""
    result: dict[int, StatementSpan] = {}
    assignment_start = re.compile(
        r"^(?:assign\s+)?[A-Za-z_][A-Za-z0-9_$\[\].:+* /()'-]*\s*(?:<=|=(?!=))"
    )
    state_encoding = re.compile(r"^[A-Z][A-Z0-9_]*\s*=.*,$")
    line_index = 0
    while line_index < len(lines):
        stripped = lines[line_index].strip()
        if (not stripped or stripped.startswith("//") or ";" in stripped
                or state_encoding.match(stripped) or not assignment_start.match(stripped)):
            line_index += 1
            continue
        end_index = line_index + 1
        while end_index < len(lines) and ";" not in lines[end_index]:
            end_index += 1
        if end_index >= len(lines) or end_index - line_index > 64:
            line_index += 1
            continue
        start = line_index + 1
        end = end_index + 1
        span = StatementSpan(start, end, _joined_statement(lines, start, end))
        for line_number in range(start, end + 1):
            result[line_number] = span
        line_index = end_index + 1
    return result


def sv_contexts(lines: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    process = "the module's structural declarations"
    process_depth = 0
    process_until: str | None = None
    state = ""
    for line_number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        starts_process = False
        if re.search(r"\balways_ff\b", stripped):
            process = "the clocked process that updates registers only on a rising clock edge"
            process_depth = max(1, len(re.findall(r"\bbegin\b", stripped)))
            process_until = None
            starts_process = True
        elif re.search(r"\balways_comb\b", stripped):
            process = "the combinational process whose outputs respond without waiting for a clock edge"
            process_depth = max(1, len(re.findall(r"\bbegin\b", stripped)))
            process_until = None
            starts_process = True
        elif stripped.startswith("function"):
            names = re.findall(r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(", stripped)
            name = names[-1] if names else "function"
            process = f"the combinational helper {humanize(name)}"
            process_until = "endfunction"
            process_depth = 0
            starts_process = True
        elif stripped.startswith("task"):
            names = re.findall(r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(", stripped)
            name = names[-1] if names else "task"
            process = f"the simulation task {humanize(name)}"
            process_until = "endtask"
            process_depth = 0
            starts_process = True
        state_match = re.match(r"(?:ST_|CORE_|WR_|RD_)?([A-Z][A-Z0-9_]*)\s*:", stripped)
        if state_match:
            state_name = state_match.group(1)
            meaning = next((value for key, value in STATE_MEANINGS.items() if key in state_name), "performs the named control phase")
            state = f" Within state {state_match.group(0)[:-1]}, the machine {meaning}."
        result[line_number] = process + state
        if stripped == "endcase":
            state = ""
        if process_until is not None and stripped.startswith(process_until):
            process = "the module's structural declarations"
            process_until = None
            process_depth = 0
            state = ""
        elif process_until is None and process != "the module's structural declarations":
            if not starts_process:
                process_depth += len(re.findall(r"\bbegin\b", stripped))
                process_depth -= len(re.findall(r"\bend\b", stripped))
            if process_depth <= 0 or (starts_process and "begin" not in stripped):
                process = "the module's structural declarations"
                process_depth = 0
                state = ""
    return result


def explain_sv_line(line: str, context: str, purpose: str) -> str:
    stripped = line.strip()
    if stripped.startswith("//"):
        comment = stripped[2:].strip()
        return f"Design intent: {comment} This comment anchors the surrounding logic in the block that {purpose}."
    if stripped.startswith("`"):
        if "default_nettype none" in stripped:
            return "This compiler directive forbids accidental implicit one-bit wires, turning misspelled signal names into errors instead of unintended hardware."
        if "timescale" in stripped:
            return "This simulation directive selects nanoseconds as the time unit and picoseconds as the timestamp precision; it does not create FPGA logic."
        return f"This SystemVerilog preprocessor line controls compilation of the block that {purpose}."
    if re.match(r"(?:module|package)\b", stripped):
        name = re.split(r"[\s(#]", stripped, maxsplit=2)[1]
        return f"This opens {name}, an independently compiled hardware definition. Its ports and statements collectively describe circuitry that {purpose}."
    if stripped.startswith(("endmodule", "endpackage")):
        return "This closes the current hardware definition; no circuit behavior is attached to the keyword itself."
    if stripped.startswith("import "):
        return f"This imports shared package names into the current module so opcodes and mode constants have exactly one definition. The import changes compile-time name visibility; it does not create registers or consume a clock cycle."
    port = re.match(r"(input|output|inout)\s+(?:wire|logic)?\s*(?:signed\s*)?(\[[^]]+\])?\s*([A-Za-z_][A-Za-z0-9_$]*)", stripped)
    if port:
        direction, width, name = port.groups()
        width_text = width or "one bit"
        return f"This declares {name} as a {direction} port with width {width_text}. It carries {identifier_meaning(name)} across the module boundary; a port is a bundle of physical FPGA nets, not a software argument."
    parameter = re.match(r"(?:parameter|localparam)\s+(?:(?:logic|integer|int|bit)\s+)?(?:signed\s*)?(?:\[[^]]+\]\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*=\s*(.*?)[,;]?$", stripped)
    if parameter:
        name, value = parameter.groups()
        return f"This fixes {name} to {value} during elaboration. Vivado uses that constant to shape or simplify the generated circuit; {name} denotes {identifier_meaning(name)}."
    if "typedef enum" in stripped:
        return "This begins a named finite-state-machine type. The encoded bits are stored in flip-flops, while symbolic names make each cycle-level phase reviewable."
    state_match = re.match(r"([A-Z][A-Z0-9_]*)\s*=\s*[^,}]+", stripped)
    if state_match:
        name = state_match.group(1)
        meaning = next((value for key, value in STATE_MEANINGS.items() if key in name), "identifies one explicit control phase")
        return f"This assigns a fixed binary encoding to state {name}, which {meaning}. The encoding is visible in waveforms even though the source uses the readable name."
    declaration = re.match(r"(?:logic|wire|reg|integer|genvar)\s+(?:signed\s*)?(?:\[[^]]+\]\s*)?(.+?);$", stripped)
    if declaration:
        names = [re.split(r"[\s[=]", item.strip(), maxsplit=1)[0] for item in declaration.group(1).split(",")]
        descriptions = "; ".join(f"{name} is {identifier_meaning(name)}" for name in names[:4])
        return f"This declares hardware signal storage or wiring in {context}. Specifically, {descriptions}. Widths in square brackets count from the stated most-significant bit down to the least-significant bit."
    assignment = re.match(r"(?:assign\s+)?([A-Za-z_][A-Za-z0-9_$\[\].:]*)\s*(<=|=)\s*(.+?);?$", stripped)
    if assignment:
        lhs, operator, rhs = assignment.groups()
        timing = "updates after the active clock edge together with other nonblocking assignments" if operator == "<=" else "is driven by combinational or procedural logic in the current evaluation"
        return f"This makes {lhs}, {identifier_meaning(lhs.split('[')[0])}, equal to `{rhs.rstrip(';')}`. It {timing}. {summarize_expression(rhs)} Context: {context}."
    if stripped.startswith("always_ff"):
        return "This opens clocked sequential logic. Everything assigned with `<=` inside becomes a flip-flop or memory write and changes at a clock edge, which creates the accelerator's notion of cycles and persistent state."
    if stripped.startswith("always_comb"):
        return "This opens combinational logic. Vivado builds gates and multiplexers whose outputs continuously follow the inputs, and the language requires every assigned output to receive a value on every path."
    if stripped.startswith("initial"):
        return "This opens simulation initialization. In a testbench it drives starting values and stimulus; project synthesis does not rely on this block as a replacement for explicit reset behavior."
    reset_match = re.match(r"if\s*\(\s*!?(?:rst_n|aresetn)\s*\)", stripped)
    if reset_match:
        return f"This is the reset branch for `{reset_match.group(0)[3:-1].strip()}`. While reset is asserted, the following assignments force control state, handshakes, counters, and error flags to known values so no stale pre-reset transaction can be mistaken for valid work. Context: {context}."
    if_match = re.match(r"(?:end\s+)?(?:else\s+)?if\s*\((.*)\)", stripped)
    if if_match:
        condition = if_match.group(1)
        return f"This condition selects the following hardware branch only when `{condition}` is true. Both branches exist as gates; the condition controls a multiplexer rather than making software jump. {summarize_expression(condition)} Context: {context}."
    if re.search(r"\b(?:unique\s+)?case\s*\((.*)\)", stripped):
        selector = re.search(r"\((.*)\)", stripped).group(1)
        return f"This starts a multiplexer/state decode selected by `{selector}`. `unique` allows tools to diagnose overlapping or missing legal choices and optimize on the expectation that one branch matches."
    label = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\s*:\s*(.*)", stripped)
    if label:
        name, action = label.groups()
        meaning = next((value for key, value in STATE_MEANINGS.items() if key in name), "performs this explicitly named phase")
        return f"This is the {name} case branch, which {meaning}. The same-cycle action begins with `{action}`; context is {context}."
    if stripped.startswith("default:"):
        return f"This is the default decode branch. It supplies deterministic behavior for every unlisted bit pattern, which prevents inferred latches in combinational logic and gives corrupted state/opcode values an explicit recovery or error path. Context: {context}."
    port_map = re.match(r"\.([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\)\s*,?$", stripped)
    if port_map:
        port_name, signal = port_map.groups()
        return f"This connects the child instance's named port `{port_name}` to parent expression `{signal}`. The connection is a physical net bundle after synthesis; direction is declared in the child module, and {identifier_meaning(signal.split('[')[0] or port_name)}."
    instance = re.match(r"([A-Za-z_][A-Za-z0-9_$]*)\s+(?:#\s*\([^;]*\)\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*\(", stripped)
    if instance and instance.group(1) not in {"if", "for", "while", "case"}:
        return f"This instantiates module {instance.group(1)} as {instance.group(2)}. Synthesis creates that subordinate circuit and connects its ports in the lines that follow."
    if stripped.startswith("for ") or stripped.startswith("for("):
        return f"This generate/procedural loop repeats the following operation across lanes or entries. In synthesizable RTL, a fixed loop is expanded into parallel or scheduled hardware rather than executed by a CPU. Context: {context}."
    if stripped in {"begin", "end", "endcase", "endgenerate", "generate"} or stripped.startswith(("end else", "else begin")):
        return f"This structural keyword opens or closes the branch/process scope described as {context}; it has no arithmetic value of its own."
    if stripped.startswith(("$display", "$fatal", "$finish", "assert")):
        return "This is a simulation check or diagnostic. It can stop a failing test and print evidence, but it is not part of the synthesized PYNQ circuit."
    return f"Within {context}, this exact statement contributes to the block that {purpose}. {summarize_expression(stripped)}"


def explain_python_line(line: str, context: str, purpose: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return f"The comment records this intent: {stripped[1:].strip()} It applies inside {context}."
    if stripped.startswith(("import ", "from ")):
        return f"This imports the named Python API so {context} can use it explicitly. Imports execute on the host processor and create no FPGA hardware."
    if stripped.startswith("class "):
        name = stripped.split()[1].split("(")[0].rstrip(":")
        return f"This defines class {name}, which groups data and methods for {humanize(name)}. Instances exist in host memory and organize the flow that {purpose}."
    if stripped.startswith(("def ", "async def ")):
        name = stripped.split("def ", 1)[1].split("(", 1)[0]
        return f"This starts function {name}. Its indented body performs {humanize(name)} as part of the file that {purpose}; it runs only when called."
    if stripped.startswith("return"):
        value = stripped.removeprefix("return").strip()
        return f"This returns `{value}` to the caller of {context}, ending this invocation. {summarize_expression(value)}"
    if stripped.startswith("raise "):
        return f"This rejects an invalid state by raising `{stripped[6:]}`. Failing before DMA prevents malformed shapes or artifacts from silently becoming hardware results."
    if stripped.startswith("assert "):
        return f"This test requires `{stripped[7:]}` to be true. A false result fails verification instead of allowing an unchecked numerical or protocol mismatch."
    if_match = re.match(r"(?:el)?if\s+(.+):$", stripped)
    if if_match:
        condition = if_match.group(1)
        return f"This executes the following indented host-code branch only when `{condition}` is true. {summarize_expression(condition)} Context: {context}."
    if stripped == "else:":
        return f"This begins the alternative host-code branch when the preceding condition was false, still inside {context}."
    loop = re.match(r"for\s+(.+?)\s+in\s+(.+):$", stripped)
    if loop:
        target, source = loop.groups()
        return f"This iterates {target} over `{source}`. Each iteration repeats the indented host operation, commonly once per layer, tile, timestep, or test case in {context}."
    if stripped.startswith("while "):
        return f"This repeats the following host-code block while `{stripped[6:-1]}` remains true; surrounding timeout/error logic prevents an unbounded hardware wait."
    if stripped.startswith("with "):
        return f"This opens a managed resource scope for `{stripped[5:-1]}` so files, tracing, or temporary state are finalized even if {context} raises an error."
    assignment = re.match(r"([A-Za-z_][A-Za-z0-9_.\[\], ]*)\s*(?::[^=]+)?\s*=\s*(.+)", stripped)
    if assignment and "==" not in stripped and ">=" not in stripped and "<=" not in stripped and "!=" not in stripped:
        lhs, rhs = assignment.groups()
        name = lhs.split(",", 1)[0].strip().split("[")[0].split(".")[-1]
        return f"This assigns `{rhs}` to `{lhs}` in host memory. Here {name} means {identifier_meaning(name)}. {summarize_expression(rhs)} Context: {context}."
    if stripped.startswith(("try:", "except ", "finally:")):
        return f"This is explicit host-side failure handling inside {context}; it ensures malformed artifacts, DMA errors, and tool failures are surfaced or cleaned up deterministically."
    if stripped.startswith("if __name__"):
        return "This standard entry-point guard runs the command-line main function only when this file is executed directly, while keeping its definitions importable by tests."
    if stripped in {"(", ")", "[", "]", "{", "}"} or stripped.endswith(("(", "[", "{")):
        return f"This line opens or closes a multi-line Python expression inside {context}; the neighboring lines supply its arguments or elements."
    if stripped.startswith("@"):
        return f"This decorator changes how the following class or function is constructed. `{stripped}` is applied at import time on the host and is part of {context}; it creates no FPGA circuit by itself."
    if stripped.startswith(('"""', "'''")):
        return f"This is executable documentation attached to {context}. It states the intended contract that tests and callers should enforce, but the text itself does not transform data or issue hardware work."
    if ".add_argument(" in stripped:
        return f"This declares a command-line option for {context}. Its name, type, default, and required status make the run configuration explicit and reproducible instead of relying on an invisible interactive setting."
    call = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)\((.*)\),?$", stripped)
    if call:
        return f"This calls `{call.group(1)}` with `{call.group(2)}` inside {context}. The call advances the host-side job that {purpose}."
    return f"Inside {context}, this statement advances the host implementation that {purpose}. {summarize_expression(stripped)}"


def explain_script_line(line: str, suffix: str, purpose: str) -> str:
    stripped = line.strip()
    if stripped.startswith("#"):
        return f"This comment documents the batch-flow intent: {stripped[1:].strip()}"
    if suffix == ".tcl":
        if stripped.startswith("read_verilog"):
            return "This loads the listed RTL into Vivado's in-memory design so the subsequent synthesis uses the repository source."
        if stripped.startswith("synth_design"):
            return "This invokes Vivado synthesis for the named top and FPGA part, mapping behavioral RTL into LUT, flip-flop, BRAM, and DSP primitives."
        if stripped.startswith(("opt_design", "place_design", "phys_opt_design", "route_design")):
            return f"This runs Vivado implementation command `{stripped}`, changing the physical netlist or placement/routing needed for real timing."
        if stripped.startswith("report_") or stripped.startswith("write_"):
            return f"This writes an auditable Vivado artifact using `{stripped}`; later documentation reads the resulting report rather than inventing a metric."
        if stripped.startswith("create_clock"):
            return "This declares the clock period against which Vivado checks every synchronous setup and hold path."
        if stripped.startswith("create_bd_cell"):
            return f"This adds the named IP or module-reference cell to the Vivado block design. The cell becomes a concrete PS, DMA, converter, interconnect, reset, or accelerator instance after generation: `{stripped}`."
        if stripped.startswith("connect_bd_intf_net"):
            return f"This connects two complete block-design interfaces, carrying all AXI channel signals defined by that interface. Vivado validates widths, directions, clocks, and protocol compatibility for `{stripped}`."
        if stripped.startswith("connect_bd_net"):
            return f"This connects scalar or bus pins in the block design. For clocks and resets, this exact connection decides the timing domain and reset island of every attached register: `{stripped}`."
        if stripped.startswith("assign_bd_address"):
            return "This asks Vivado to establish an AXI address map; following explicit offset/range settings make the software-visible register addresses stable across rebuilds."
        if stripped.startswith(("launch_runs", "wait_on_run", "open_run")):
            return f"This controls a named Vivado synthesis or implementation run with `{stripped}`. Waiting and opening the completed run prevents reports from reading an earlier or unfinished netlist."
        if stripped.startswith("read_saif"):
            return f"This annotates implemented nets with measured XSim switching activity from the named SAIF file. The retained match report quantifies which simulated names mapped to routed nets: `{stripped}`."
    else:
        if "$ErrorActionPreference" in stripped:
            return "This makes PowerShell stop on errors so a failed prerequisite cannot be hidden by later successful commands."
        if stripped.startswith("param") or re.match(r"\[[A-Za-z]+\]\$", stripped):
            return "This declares a typed command-line parameter so the flow can be rerun with explicit, validated paths and options."
        if "LASTEXITCODE" in stripped:
            return "This checks the external program's process exit code and turns a failed compiler, simulator, or Vivado run into a failed pipeline."
        if re.search(r"vivado|xsim|xelab|xvlog|iverilog|verilator|python", stripped, re.IGNORECASE):
            return f"This locates or invokes a required tool as part of the script that {purpose}; the exact command and arguments remain visible here for reproducibility."
        if stripped.startswith(("Resolve-Path", "Join-Path", "$ProjectRoot")) or "[System.IO.Path]::GetFullPath" in stripped:
            return f"This resolves a deterministic filesystem location for the flow that {purpose}. Literal/full paths prevent a different working directory from silently selecting the wrong source, checkpoint, or output."
        if stripped.startswith("New-Item"):
            return "This creates the declared output directory before a tool writes into it. `-Force` permits an existing directory but does not make an external tool failure succeed."
        if stripped.startswith("Remove-Item"):
            return "This removes the explicitly named rebuildable artifact. The script uses a literal path so wildcard expansion cannot delete an unintended neighboring file."
    return f"This batch-control line contributes to the flow that {purpose}. {summarize_expression(stripped)}"


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}",
        ">": r"\textgreater{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def source_architecture_notes(entry: SourceEntry, text: str) -> list[tuple[str, str]]:
    """Build a file-level map so line annotations have an understandable frame."""
    suffix = entry.path.suffix.lower()
    dataflow = FILE_DATAFLOWS.get(
        entry.relative,
        f"This file's end-to-end responsibility is to {entry.purpose}.",
    )
    notes: list[tuple[str, str]] = [("Dataflow responsibility", dataflow)]
    if suffix in {".sv", ".v"} and not entry.relative.startswith("tb/"):
        modules = re.findall(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", text, re.MULTILINE)
        inputs = len(re.findall(r"^\s*input\b", text, re.MULTILINE))
        outputs = len(re.findall(r"^\s*output\b", text, re.MULTILINE))
        clocked = len(re.findall(r"\balways_ff\b", text))
        combinational = len(re.findall(r"\balways_comb\b", text))
        states = sorted(set(re.findall(r"\bST_[A-Z0-9_]+\b", text)))
        instances: list[str] = []
        instance_pattern = re.compile(
            r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+(?:#\s*\([^;]*\)\s*)?"
            r"([A-Za-z_][A-Za-z0-9_$]*)\s*\(", re.MULTILINE,
        )
        for module_name, instance_name in instance_pattern.findall(text):
            if module_name in {"module", "function", "task", "if", "for", "while", "case"}:
                continue
            label = f"{module_name} as {instance_name}"
            if label not in instances:
                instances.append(label)
        notes.extend(
            [
                (
                    "What becomes FPGA hardware",
                    f"Vivado elaborates module(s) {', '.join(modules) or 'declared here'} into physical nets and primitives. The source declares {inputs} input-port lines, {outputs} output-port lines, {clocked} clocked process(es), and {combinational} combinational process(es). Clocked assignments become retained state; combinational assignments become gates, arithmetic, and multiplexers.",
                ),
                (
                    "Cycle-level control",
                    (
                        f"The named control phases include {', '.join(states[:18])}"
                        + (" and additional states" if len(states) > 18 else "")
                        + ". A state is a binary value held in flip-flops. On each rising edge, accepted valid/ready handshakes and current state determine the next state and which memories or output registers update."
                        if states
                        else "This block is primarily a datapath or storage primitive rather than a large state machine. Registers still divide combinational work into cycles and hold valid/data stable during downstream stalls."
                    ),
                ),
                (
                    "Connections to other blocks",
                    (
                        "This source instantiates " + ", ".join(instances[:12]) + ". Port-map lines connect actual parent wires to each child's named ports; no data is copied merely by naming a connection."
                        if instances
                        else "This is a leaf block: parent modules create it and connect its ports. It does not instantiate another project RTL module, so its arithmetic or storage maps directly into FPGA primitives."
                    ),
                ),
                (
                    "How to debug it",
                    "In a waveform, place clock and reset first, then each interface's valid, ready, data, and last signals together. A transfer exists only on an edge where valid and ready are both one. Add state, address/index, accumulator, and error signals to explain why the interface advanced or stalled.",
                ),
            ]
        )
    elif entry.relative.startswith("tb/"):
        tops = re.findall(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", text, re.MULTILINE)
        checks = len(re.findall(r"\$(?:fatal|error)|\bassert\b|FAILED|PASSED", text))
        notes.extend(
            [
                ("Simulation-only status", f"{', '.join(tops) or 'This testbench'} is not synthesized into the FPGA. It generates clocks, reset, commands, stalls, and expected values around the design under test, then terminates with an explicit result marker."),
                ("Evidence produced", f"The file contains approximately {checks} explicit assertions, failure diagnostics, or pass/fail markers. Its VCD/WDB trace is useful evidence only when the final pass marker is present; a simulator exit code by itself is insufficient."),
            ]
        )
    elif suffix == ".py":
        try:
            tree = ast.parse(text)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        except SyntaxError:
            classes, functions = [], []
        notes.extend(
            [
                ("Where it runs", "This is sequential host software. Python objects live in CPU memory; they become FPGA work only when the compiler packs a command or payload and the PYNQ driver transfers those bytes through AXI DMA or AXI-Lite."),
                ("Public structure", f"Classes: {', '.join(classes[:16]) or 'none'}. Functions: {', '.join(functions[:24]) or 'none'}. Each function annotation below names its enclosing function so an array operation is tied to the model, compiler, driver, test, or report phase that uses it."),
                ("Numeric and failure boundary", "Shapes, signedness, byte order, scaling, and saturation are part of the hardware contract. Validation exceptions are functional safety checks: rejecting malformed input before DMA is different from silently clipping, reshaping, or continuing with an incompatible artifact."),
            ]
        )
    else:
        tool = "Vivado Tcl" if suffix == ".tcl" else "PowerShell"
        notes.extend(
            [
                ("Where it runs", f"This is {tool} orchestration, not accelerator datapath hardware. It makes tool inputs, versions, paths, stages, and failure propagation repeatable."),
                ("Reproducibility contract", "Every external command must be checked, every required artifact must exist and be nonempty, and measured values must be read from generated reports. Rerunning this file should either reproduce its declared outputs or stop at the first failed gate."),
            ]
        )
    return notes


def annotation_tex(entry: SourceEntry, text: str) -> tuple[list[str], int]:
    lines = text.splitlines()
    suffix = entry.path.suffix.lower()
    contexts = python_contexts(text) if suffix == ".py" else sv_contexts(lines) if suffix in {".sv", ".v"} else {}
    statement_spans = (
        python_statement_spans(text) if suffix == ".py"
        else sv_statement_spans(lines) if suffix in {".sv", ".v"}
        else {}
    )
    result: list[str] = []
    explained = 0
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        context = contexts.get(line_number, f"the file whose role is to {entry.purpose}")
        span = statement_spans.get(line_number)
        if suffix in {".sv", ".v"}:
            complete_explanation = explain_sv_line(
                span.text if span else raw, context, entry.purpose)
        elif suffix == ".py":
            complete_explanation = explain_python_line(
                span.text if span else raw, context, entry.purpose)
        else:
            complete_explanation = explain_script_line(raw, suffix, entry.purpose)
        if span and line_number != span.start:
            position = line_number - span.start + 1
            length = span.end - span.start + 1
            explanation = (
                f"This is continuation line {position} of the {length}-line logical statement "
                f"that begins at source line {span.start}. Its exact fragment is `{raw.strip()}`. "
                f"{summarize_expression(raw)} Read together, the complete operation is: "
                f"{complete_explanation}"
            )
        elif span:
            explanation = (
                f"This statement continues through source line {span.end}; the explanation uses "
                f"the complete expression rather than only the first visual fragment. "
                f"{complete_explanation}"
            )
        else:
            explanation = complete_explanation
        preview = raw.strip()
        if len(preview) > 110:
            preview = preview[:107] + "..."
        result.extend(
            [
                rf"\paragraph{{Line {line_number}: \texttt{{{latex_escape(preview)}}}}}",
                latex_escape(explanation),
            ]
        )
        explained += 1
    return result, explained


def generate(output_dir: Path) -> tuple[Path, Path]:
    root = project_root()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = source_entries(root)
    tex_path = output_dir / "neuraedge_line_by_line_source_reference.tex"
    manifest_path = output_dir / "neuraedge_line_by_line_source_manifest.json"
    tex = [
        r"\documentclass[9pt]{article}",
        r"\usepackage[margin=0.62in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{xcolor}",
        r"\usepackage{listings}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{3pt}",
        r"\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,keepspaces=true,showstringspaces=false,numbers=left,numberstyle=\tiny,frame=single}",
        r"\title{NeuraEdge Universal: Complete Line-by-Line Source Reference}",
        r"\author{Generated from hashed repository source}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        r"\section{How to read this volume}",
        "Each file begins with its system-level responsibility and a complete numbered source listing. The following annotations explain every nonblank line using its enclosing function, RTL process, or finite-state-machine phase. Blank lines are visible in the listing but have no executable or synthesized behavior. SystemVerilog describes parallel hardware: an if statement controls a multiplexer and a nonblocking assignment updates a register on a clock edge. Python, PowerShell, and Tcl run sequentially on a processor or inside a tool.",
    ]
    manifest: dict[str, object] = {"format": "neuraedge-source-reference-v1", "files": []}
    total_lines = 0
    total_nonblank = 0
    for entry in entries:
        raw_bytes = entry.path.read_bytes()
        text = raw_bytes.decode("utf-8", errors="strict")
        line_count = len(text.splitlines())
        nonblank_count = sum(bool(line.strip()) for line in text.splitlines())
        annotations, explained = annotation_tex(entry, text)
        if explained != nonblank_count:
            raise RuntimeError(f"annotation coverage mismatch for {entry.relative}: {explained} != {nonblank_count}")
        total_lines += line_count
        total_nonblank += nonblank_count
        listing_path = os.path.relpath(entry.path, output_dir).replace("\\", "/")
        manifest["files"].append(
            {
                "path": entry.relative,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "lines": line_count,
                "nonblank_lines": nonblank_count,
                "explained_nonblank_lines": explained,
                "purpose": entry.purpose,
            }
        )
        tex.extend(
            [
                r"\clearpage",
                rf"\section{{\texttt{{{latex_escape(entry.relative)}}}}}",
                rf"\textbf{{Role in the complete system:}} {latex_escape(entry.purpose)}.",
                r"\subsection{Beginner architecture map}",
                r"\begin{description}",
                *[
                    rf"\item[{latex_escape(label)}] {latex_escape(note)}"
                    for label, note in source_architecture_notes(entry, text)
                ],
                r"\end{description}",
                r"\subsection{Complete source}",
                rf"\lstinputlisting{{{listing_path}}}",
                r"\subsection{Line annotations}",
                *annotations,
            ]
        )
    manifest["summary"] = {
        "file_count": len(entries),
        "source_lines": total_lines,
        "nonblank_lines": total_nonblank,
        "explained_nonblank_lines": total_nonblank,
    }
    tex.extend(
        [
            r"\clearpage",
            r"\section{Coverage statement}",
            f"This volume contains {len(entries)} source files, {total_lines:,} physical source lines, and an explanation for all {total_nonblank:,} nonblank lines. The companion JSON manifest records each source SHA-256 digest.",
            r"\end{document}",
        ]
    )
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return tex_path, manifest_path


def compile_pdf(tex_path: Path, pdflatex: str) -> Path:
    for _ in range(2):
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent,
            check=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=project_root() / "docs")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pdflatex", default="pdflatex")
    args = parser.parse_args()
    tex_path, manifest_path = generate(args.output_dir.resolve())
    print(f"tex={tex_path}")
    print(f"manifest={manifest_path}")
    if args.compile:
        print(f"pdf={compile_pdf(tex_path, args.pdflatex)}")


if __name__ == "__main__":
    main()
