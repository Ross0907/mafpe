#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

sources=(
    rtl/neuraedge_command_pkg.sv
    rtl/neuraedge_mac2_u8.sv
    rtl/neuraedge_tensor_tile.sv
    rtl/neuraedge_active_index_buffer.sv
    rtl/neuraedge_activation_buffer.sv
    rtl/neuraedge_weight_bank.sv
    rtl/neuraedge_buffered_tensor_core.sv
    rtl/neuraedge_alpha_xnor.sv
    rtl/neuraedge_vector_alu.sv
    rtl/neuraedge_udivider.sv
    rtl/neuraedge_isqrt64.sv
    rtl/neuraedge_integer_layernorm.sv
    rtl/neuraedge_log_softmax.sv
    rtl/neuraedge_lif_bank.sv
    rtl/neuraedge_sgd_optimizer8.sv
    rtl/neuraedge_gradient_accumulator_bank.sv
    rtl/neuraedge_linear_backward8.sv
    rtl/neuraedge_activation_lut8.sv
    rtl/neuraedge_universal_core.sv
    rtl/neuraedge_descriptor_sequencer.sv
    rtl/neuraedge_pynq_axi_top.sv
)

mkdir -p outputs/regression
verilator \
    --lint-only \
    --timing \
    -Wall \
    --top-module neuraedge_pynq_axi_top \
    -I. \
    config/verilator.vlt \
    "${sources[@]}" \
    2>&1 | tee outputs/regression/verilator_lint.log
