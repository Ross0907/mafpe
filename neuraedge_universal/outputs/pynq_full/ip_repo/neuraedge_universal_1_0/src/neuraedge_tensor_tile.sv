`timescale 1ns / 1ps
`default_nettype none

module neuraedge_tensor_tile #(
    parameter int ROW_LANES = 8,
    parameter int COLUMN_LANES = 16,
    parameter int ACC_BITS = 32
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire [1:0] compute_mode,
    input  wire [3:0] active_rows,
    input  wire [5:0] active_columns,
    input  wire [7:0] activation_zero_point,
    input  wire [7:0] weight_zero_point,
    output logic config_error,

    input  wire issue_valid,
    output logic issue_ready,
    input  wire [ROW_LANES*8-1:0] activation_vector,
    input  wire [COLUMN_LANES*8-1:0] outer_weight_vector,
    input  wire [COLUMN_LANES*ROW_LANES*8-1:0] dot_weight_vector,
    input  wire issue_last,

    output logic busy,
    output logic result_valid,
    input  wire result_ready,
    output logic signed [47:0] result_value,
    output logic [3:0] result_row,
    output logic [5:0] result_column,
    output logic result_last,

    output logic [31:0] debug_accepted_issues,
    output logic [31:0] debug_skipped_issues,
    output logic [31:0] debug_retired_issues,
    output logic [63:0] debug_dense_macs,
    output logic [63:0] debug_spike_adds
);
    localparam logic [1:0] MODE_DENSE_OUTER = 2'd0;
    localparam logic [1:0] MODE_DENSE_DOT8 = 2'd1;
    localparam logic [1:0] MODE_BINARY_WEIGHTED = 2'd2;
    localparam logic [1:0] MODE_TERNARY_WEIGHTED = 2'd3;
    localparam int COLUMN_PAIRS = COLUMN_LANES / 2;
    localparam int RESULT_FIFO_DEPTH = ROW_LANES * COLUMN_LANES;
    localparam int RESULT_FIFO_ADDRESS_BITS = $clog2(RESULT_FIFO_DEPTH);
    localparam int RESULT_FIFO_COUNT_BITS = $clog2(RESULT_FIFO_DEPTH + 1);

    typedef enum logic [3:0] {
        ST_IDLE = 4'd0,
        ST_RUN = 4'd1,
        ST_DRAIN = 4'd2,
        ST_PREPARE_ZERO_POINT = 4'd3,
        ST_RESTORE_MULTIPLY = 4'd4,
        ST_OUTPUT_PIPELINE = 4'd5,
        ST_LATCH_ZERO_POINT = 4'd6,
        ST_RESTORE_ADD = 4'd7,
        ST_SPARSE_DRAIN = 4'd8,
        ST_CAPTURE_TARGET = 4'd9,
        ST_COMBINE_ZERO_POINT = 4'd10,
        ST_PREPARE_OUTPUT_COUNT = 4'd11,
        ST_ACCUMULATE_OUTPUT_COUNT = 4'd12,
        ST_RESTORE_RELAY = 4'd13
    } state_t;

    state_t state;
    (* dont_touch = "yes" *) logic [1:0] mode_r;
    (* dont_touch = "yes" *) logic [3:0] rows_r;
    logic [ROW_LANES-1:0] active_row_mask_r;
    logic [5:0] columns_r;
    logic [7:0] activation_zp_r;
    logic [7:0] weight_zp_r;

    logic signed [ACC_BITS-1:0] accumulator [0:ROW_LANES-1][0:COLUMN_LANES-1];
    // Keep one physical clear source per row. A command-start state bit driving
    // every accumulator bit creates a long, high-fanout control path after
    // placement; these registered row-local pulses perform the same clear one
    // cycle after start, before the issue pipeline can produce a delta.
    (* dont_touch = "yes", max_fanout = 96 *)
    logic accumulator_clear_row_r [0:ROW_LANES-1];
    // Column sums use the same delayed-clear scheme. One register per column
    // keeps the command-start decode out of a 512-bit distributed CE network.
    (* dont_touch = "yes", max_fanout = 48 *)
    logic column_sum_clear_r [0:COLUMN_LANES-1];
    (* dont_touch = "yes" *) logic accumulate_delta_valid_r;
    logic signed [20:0] accumulate_delta_r [0:ROW_LANES-1][0:COLUMN_LANES-1];
    logic [31:0] row_sum [0:ROW_LANES-1];
    logic [31:0] column_sum [0:COLUMN_LANES-1];
    logic sum_update_valid_r;
    logic [15:0] row_increment_r [0:ROW_LANES-1];
    logic [15:0] column_increment_r [0:COLUMN_LANES-1];
    logic [31:0] effective_reduction;

    logic [11:0] mac_issue_count;
    // The counter is updated in both run and command-boundary states.  A local
    // D-input hold avoids a long FSM-to-CE route across the MAC array.
    (* extract_enable = "no" *) logic [11:0] mac_retire_count;
    logic [11:0] target_mac_retires;
    // One command can execute at most 2048*8*16 = 262144 useful operations,
    // so 19 bits represent the exact per-command telemetry range. The
    // increment is registered before accumulation to keep popcount and lane
    // sizing logic out of the counter carry chain at 200 MHz.
    logic [18:0] debug_dense_macs_r;
    logic [18:0] debug_spike_adds_r;
    logic [7:0] metric_increment_r;
    logic metric_increment_valid_r;
    logic [3:0] sparse_active_count_r;
    logic sparse_metric_valid_r;
    logic [7:0] dense_macs_per_issue_r;
    (* dont_touch = "yes" *) logic drain_complete_r;
    (* use_dsp = "yes" *) logic [15:0] zero_point_product_r;
    logic [31:0] zero_restore_reduction_operand_r;
    logic [15:0] zero_restore_zp_operand_r;
    (* use_dsp = "yes" *) logic [31:0] zero_restore_low_product_r;
    (* use_dsp = "yes" *) logic [31:0] zero_restore_high_product_r;
    logic [31:0] zero_restore_low_relay_r;
    logic [31:0] zero_restore_high_relay_r;
    logic [47:0] zero_point_restore_r;
    logic [7:0] launch_total_r;
    (* extract_enable = "no" *) logic [7:0] launch_remaining_r;
    logic [3:0] launch_rows_remaining_r;
    (* extract_enable = "no" *) logic [5:0] launch_columns_remaining_r;
    (* extract_enable = "no" *) logic [3:0] launch_row_r;
    (* extract_enable = "no" *) logic [5:0] launch_column_r;
    logic pread_valid_r;
    logic signed [ACC_BITS-1:0] pread_accumulator_by_row_r [0:ROW_LANES-1];
    logic [31:0] pread_row_sum_r;
    logic [31:0] pread_column_sum_r;
    logic [3:0] pread_row_r;
    logic [5:0] pread_column_r;
    (* dont_touch = "yes" *) logic p0_valid_r;
    logic signed [63:0] p0_raw_r;
    logic [31:0] p0_row_sum_r;
    logic [31:0] p0_column_sum_r;
    logic [7:0] p0_row_zp_low_r;
    logic [7:0] p0_row_zp_high_r;
    logic [7:0] p0_column_zp_low_r;
    logic [7:0] p0_column_zp_high_r;
    logic [3:0] p0_row_r;
    logic [5:0] p0_column_r;
    (* dont_touch = "yes" *) logic p1_valid_r;
    logic signed [63:0] p1_raw_r;
    logic [23:0] p1_row_product_low_r;
    logic [23:0] p1_row_product_high_r;
    logic [23:0] p1_column_product_low_r;
    logic [23:0] p1_column_product_high_r;
    logic [3:0] p1_row_r;
    logic [5:0] p1_column_r;
    (* dont_touch = "yes" *) logic p2_valid_r;
    logic signed [63:0] p2_raw_r;
    logic signed [63:0] p2_row_correction_r;
    logic signed [63:0] p2_column_correction_r;
    logic [3:0] p2_row_r;
    logic [5:0] p2_column_r;
    (* dont_touch = "yes" *) logic p3_valid_r;
    logic signed [63:0] p3_corrected_r;
    logic signed [63:0] p3_column_correction_r;
    logic [3:0] p3_row_r;
    logic [5:0] p3_column_r;
    (* dont_touch = "yes" *) logic p4_valid_r;
    logic signed [63:0] p4_corrected_r;
    logic [3:0] p4_row_r;
    logic [5:0] p4_column_r;
    (* dont_touch = "yes" *) logic p5_valid_r;
    logic signed [47:0] p5_result_r;
    logic [3:0] p5_row_r;
    logic [5:0] p5_column_r;
    logic output_pipeline_enable;
    logic [RESULT_FIFO_COUNT_BITS-1:0] result_fifo_count_r;
    (* dont_touch = "yes" *) logic result_fifo_has_space_r;
    (* dont_touch = "yes" *) logic result_fifo_nonempty_r;
    logic [RESULT_FIFO_ADDRESS_BITS-1:0] result_fifo_write_pointer_r;
    logic [RESULT_FIFO_ADDRESS_BITS-1:0] result_fifo_read_pointer_r;
    logic [RESULT_FIFO_ADDRESS_BITS-1:0] result_fifo_last_address_r;
    // One command emits at most 8*16 records. Buffering that complete tile
    // removes downstream backpressure from every arithmetic pipeline CE while
    // retaining an asynchronous read suitable for a stable AXI-Stream output.
    (* ram_style = "distributed" *) logic [57:0] result_fifo_payload_r
        [0:RESULT_FIFO_DEPTH-1];
    logic result_stage_valid_r;
    logic [57:0] result_stage_payload_r;
    logic result_stage_last_r;
    wire result_fifo_push = state == ST_OUTPUT_PIPELINE
        && p5_valid_r && result_fifo_has_space_r;
    wire result_fifo_pop = state == ST_OUTPUT_PIPELINE
        && result_fifo_nonempty_r
        && (!result_stage_valid_r
            || (result_stage_valid_r && result_ready));
    wire result_stage_pop = state == ST_OUTPUT_PIPELINE
        && result_stage_valid_r && result_ready;

    // Zero-point restoration is computed once per tile. Row and column products
    // are independent and therefore form a throughput-one correction pipeline.
    (* use_dsp = "yes" *) wire [23:0] p0_row_product_low =
        p0_row_sum_r[15:0] * p0_row_zp_low_r;
    (* use_dsp = "yes" *) wire [23:0] p0_row_product_high =
        p0_row_sum_r[31:16] * p0_row_zp_high_r;
    (* use_dsp = "yes" *) wire [23:0] p0_column_product_low =
        p0_column_sum_r[15:0] * p0_column_zp_low_r;
    (* use_dsp = "yes" *) wire [23:0] p0_column_product_high =
        p0_column_sum_r[31:16] * p0_column_zp_high_r;

    logic [7:0] mac_activation [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    logic [7:0] mac_weight_lo [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    logic [7:0] mac_weight_hi [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    wire [15:0] mac_product_lo [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    wire [15:0] mac_product_hi [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    wire mac_product_valid [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    logic dot_pair_valid_r;
    logic dot_half_valid_r;
    logic dot_sum_valid_r;
    logic [16:0] dot_pair_lo_r [0:COLUMN_PAIRS-1][0:3];
    logic [16:0] dot_pair_hi_r [0:COLUMN_PAIRS-1][0:3];
    logic [17:0] dot_half_lo_r [0:COLUMN_PAIRS-1][0:1];
    logic [17:0] dot_half_hi_r [0:COLUMN_PAIRS-1][0:1];
    logic [19:0] dot_sum_lo_r [0:COLUMN_PAIRS-1];
    logic [19:0] dot_sum_hi_r [0:COLUMN_PAIRS-1];
    logic dot8_pair_valid_r;
    logic [8:0] dot8_row_pair_r [0:3];
    // As with the retire counter, keep the classify-state hold on the data
    // input so each pair register can be implemented near its adder.
    (* extract_enable = "no" *)
    logic [8:0] dot8_column_pair_r [0:COLUMN_LANES-1][0:3];
    logic dot8_increment_half_valid_r;
    logic [9:0] dot8_row_half_r [0:1];
    logic [9:0] dot8_column_half_r [0:COLUMN_LANES-1][0:1];

    logic issue_accept;
    logic input_done_r;
    logic issue_match_valid_r;
    logic [7:0] issue_zero_matches_r;
    logic issue_match_last_r;
    logic [ROW_LANES*8-1:0] issue_match_activation_vector_r;
    logic [COLUMN_LANES*8-1:0] issue_match_outer_weight_vector_r;
    logic [COLUMN_LANES*ROW_LANES*8-1:0] issue_match_dot_weight_vector_r;
    logic [7:0] issue_zero_matches_comb;

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [1:0] pair0;
        logic [1:0] pair1;
        logic [1:0] pair2;
        logic [1:0] pair3;
        logic [2:0] half0;
        logic [2:0] half1;
        begin
            pair0 = value[0] + value[1];
            pair1 = value[2] + value[3];
            pair2 = value[4] + value[5];
            pair3 = value[6] + value[7];
            half0 = pair0 + pair1;
            half1 = pair2 + pair3;
            popcount8 = half0 + half1;
        end
    endfunction
    (* dont_touch = "yes" *) logic classify_valid_r;
    (* dont_touch = "yes" *) logic classify_is_zero_r;
    logic classify_last_r;
    logic [ROW_LANES*8-1:0] classify_activation_vector_r;
    logic [COLUMN_LANES*8-1:0] classify_outer_weight_vector_r;
    logic [COLUMN_LANES*ROW_LANES*8-1:0] classify_dot_weight_vector_r;
    logic dense_issue_enable;
    (* dont_touch = "yes" *) logic dense_issue_enable_r;
    logic [ROW_LANES*8-1:0] dense_activation_vector_r;
    logic [7:0] dense_weight_lo_r [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    logic [7:0] dense_weight_hi_r [0:ROW_LANES-1][0:COLUMN_PAIRS-1];
    logic dense_outer_tag0_r;
    logic dense_outer_tag1_r;
    logic dense_outer_tag2_r;
    logic sparse_prepare_valid_r;
    logic [ROW_LANES-1:0] sparse_prepare_active_r;
    logic [ROW_LANES-1:0] sparse_prepare_negative_r;
    logic [COLUMN_LANES*8-1:0] sparse_prepare_weight_vector_r;
    (* dont_touch = "yes" *) logic sparse_delta_valid_r;
    logic signed [8:0] sparse_delta_r [0:ROW_LANES-1][0:COLUMN_LANES-1];
    wire dense_product_valid = mac_product_valid[0][0];
    wire dense_accumulate_valid = dense_outer_tag2_r || dot_sum_valid_r;

    function automatic logic [15:0] sum_u8x8(input logic [63:0] values);
        logic [8:0] pair0;
        logic [8:0] pair1;
        logic [8:0] pair2;
        logic [8:0] pair3;
        logic [9:0] half0;
        logic [9:0] half1;
        begin
            pair0 = values[7:0] + values[15:8];
            pair1 = values[23:16] + values[31:24];
            pair2 = values[39:32] + values[47:40];
            pair3 = values[55:48] + values[63:56];
            half0 = pair0 + pair1;
            half1 = pair2 + pair3;
            sum_u8x8 = {6'd0, half0} + {6'd0, half1};
        end
    endfunction

    function automatic logic [19:0] sum_u16x8(
        input logic [15:0] value0,
        input logic [15:0] value1,
        input logic [15:0] value2,
        input logic [15:0] value3,
        input logic [15:0] value4,
        input logic [15:0] value5,
        input logic [15:0] value6,
        input logic [15:0] value7
    );
        logic [16:0] pair0;
        logic [16:0] pair1;
        logic [16:0] pair2;
        logic [16:0] pair3;
        logic [17:0] half0;
        logic [17:0] half1;
        begin
            pair0 = value0 + value1;
            pair1 = value2 + value3;
            pair2 = value4 + value5;
            pair3 = value6 + value7;
            half0 = pair0 + pair1;
            half1 = pair2 + pair3;
            sum_u16x8 = {2'd0, half0} + {2'd0, half1};
        end
    endfunction

    always_comb begin
        for (int lane = 0; lane < ROW_LANES; lane++) begin
            unique case (mode_r)
                MODE_DENSE_OUTER,
                MODE_DENSE_DOT8: issue_zero_matches_comb[lane] =
                    activation_vector[lane*8 +: 8] == activation_zp_r;
                MODE_BINARY_WEIGHTED,
                MODE_TERNARY_WEIGHTED: issue_zero_matches_comb[lane] =
                    activation_vector[lane*8 +: 8] == 8'd0;
                default: issue_zero_matches_comb[lane] = 1'b1;
            endcase
        end
    end

    assign issue_accept = issue_valid && issue_ready;
    assign dense_issue_enable = dense_issue_enable_r;

    always_comb begin
        for (int lane = 0; lane < ROW_LANES; lane++) begin
            for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                mac_activation[lane][pair_idx] =
                    dense_activation_vector_r[lane*8 +: 8];
                mac_weight_lo[lane][pair_idx] = dense_weight_lo_r[lane][pair_idx];
                mac_weight_hi[lane][pair_idx] = dense_weight_hi_r[lane][pair_idx];
            end
        end
    end

    for (genvar lane = 0; lane < ROW_LANES; lane++) begin : g_mac_lane
        for (genvar pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin : g_mac_pair
            neuraedge_mac2_u8 u_mac2 (
                .clk(clk),
                .rst_n(rst_n),
                .enable(dense_issue_enable),
                .activation(mac_activation[lane][pair_idx]),
                .weight_lo(mac_weight_lo[lane][pair_idx]),
                .weight_hi(mac_weight_hi[lane][pair_idx]),
                .product_valid(mac_product_valid[lane][pair_idx]),
                .product_lo(mac_product_lo[lane][pair_idx]),
                .product_hi(mac_product_hi[lane][pair_idx])
            );
        end
    end

    always_comb begin
        issue_ready = (state == ST_RUN) && !input_done_r;
        busy = (state != ST_IDLE);
        result_valid = (state == ST_OUTPUT_PIPELINE) && result_stage_valid_r;
        result_row = result_stage_payload_r[51:48];
        result_column = result_stage_payload_r[57:52];
        result_last = result_stage_last_r;
        result_value = result_stage_payload_r[47:0];
        debug_dense_macs = {45'd0, debug_dense_macs_r};
        debug_spike_adds = {45'd0, debug_spike_adds_r};
        // The FIFO holds the largest legal tile, so all arithmetic stages can
        // advance every cycle regardless of downstream READY.
        output_pipeline_enable = 1'b1;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            mode_r <= MODE_DENSE_OUTER;
            rows_r <= 4'd1;
            active_row_mask_r <= {{(ROW_LANES-1){1'b0}}, 1'b1};
            columns_r <= 6'd1;
            activation_zp_r <= 8'd0;
            weight_zp_r <= 8'd0;
            config_error <= 1'b0;
            effective_reduction <= 32'd0;
            mac_issue_count <= 12'd0;
            mac_retire_count <= 12'd0;
            target_mac_retires <= 12'd0;
            drain_complete_r <= 1'b0;
            zero_point_product_r <= 16'd0;
            zero_restore_reduction_operand_r <= 32'd0;
            zero_restore_zp_operand_r <= 16'd0;
            zero_restore_low_product_r <= 32'd0;
            zero_restore_high_product_r <= 32'd0;
            zero_restore_low_relay_r <= 32'd0;
            zero_restore_high_relay_r <= 32'd0;
            zero_point_restore_r <= 48'd0;
            launch_total_r <= 8'd0;
            launch_remaining_r <= 8'd0;
            launch_rows_remaining_r <= 4'd0;
            launch_columns_remaining_r <= 6'd0;
            launch_row_r <= 4'd0;
            launch_column_r <= 6'd0;
            pread_valid_r <= 1'b0;
            pread_row_sum_r <= 32'd0;
            pread_column_sum_r <= 32'd0;
            pread_row_r <= 4'd0;
            pread_column_r <= 6'd0;
            p0_valid_r <= 1'b0;
            p0_raw_r <= 64'sd0;
            p0_row_sum_r <= 32'd0;
            p0_column_sum_r <= 32'd0;
            p0_row_zp_low_r <= 8'd0;
            p0_row_zp_high_r <= 8'd0;
            p0_column_zp_low_r <= 8'd0;
            p0_column_zp_high_r <= 8'd0;
            p0_row_r <= 4'd0;
            p0_column_r <= 6'd0;
            p1_valid_r <= 1'b0;
            p1_raw_r <= 64'sd0;
            p1_row_product_low_r <= 24'd0;
            p1_row_product_high_r <= 24'd0;
            p1_column_product_low_r <= 24'd0;
            p1_column_product_high_r <= 24'd0;
            p1_row_r <= 4'd0;
            p1_column_r <= 6'd0;
            p2_valid_r <= 1'b0;
            p2_raw_r <= 64'sd0;
            p2_row_correction_r <= 64'sd0;
            p2_column_correction_r <= 64'sd0;
            p2_row_r <= 4'd0;
            p2_column_r <= 6'd0;
            p3_valid_r <= 1'b0;
            p3_corrected_r <= 64'sd0;
            p3_column_correction_r <= 64'sd0;
            p3_row_r <= 4'd0;
            p3_column_r <= 6'd0;
            p4_valid_r <= 1'b0;
            p4_corrected_r <= 64'sd0;
            p4_row_r <= 4'd0;
            p4_column_r <= 6'd0;
            p5_valid_r <= 1'b0;
            p5_result_r <= 48'sd0;
            p5_row_r <= 4'd0;
            p5_column_r <= 6'd0;
            result_fifo_count_r <= '0;
            result_fifo_has_space_r <= 1'b1;
            result_fifo_nonempty_r <= 1'b0;
            result_fifo_write_pointer_r <= '0;
            result_fifo_read_pointer_r <= '0;
            result_fifo_last_address_r <= '0;
            result_stage_valid_r <= 1'b0;
            result_stage_payload_r <= '0;
            result_stage_last_r <= 1'b0;
            accumulate_delta_valid_r <= 1'b0;
            debug_accepted_issues <= 32'd0;
            debug_skipped_issues <= 32'd0;
            debug_retired_issues <= 32'd0;
            debug_dense_macs_r <= 19'd0;
            debug_spike_adds_r <= 19'd0;
            metric_increment_r <= 8'd0;
            metric_increment_valid_r <= 1'b0;
            sparse_active_count_r <= 4'd0;
            sparse_metric_valid_r <= 1'b0;
            dense_macs_per_issue_r <= 8'd0;
            dot_sum_valid_r <= 1'b0;
            dot_pair_valid_r <= 1'b0;
            dot_half_valid_r <= 1'b0;
            dot8_pair_valid_r <= 1'b0;
            dot8_increment_half_valid_r <= 1'b0;
            sum_update_valid_r <= 1'b0;
            input_done_r <= 1'b0;
            issue_match_valid_r <= 1'b0;
            issue_zero_matches_r <= 8'd0;
            issue_match_last_r <= 1'b0;
            issue_match_activation_vector_r <= '0;
            issue_match_outer_weight_vector_r <= '0;
            issue_match_dot_weight_vector_r <= '0;
            classify_valid_r <= 1'b0;
            classify_is_zero_r <= 1'b0;
            classify_last_r <= 1'b0;
            classify_activation_vector_r <= '0;
            classify_outer_weight_vector_r <= '0;
            classify_dot_weight_vector_r <= '0;
            dense_issue_enable_r <= 1'b0;
            dense_activation_vector_r <= '0;
            dense_outer_tag0_r <= 1'b0;
            dense_outer_tag1_r <= 1'b0;
            dense_outer_tag2_r <= 1'b0;
            sparse_prepare_valid_r <= 1'b0;
            sparse_prepare_active_r <= '0;
            sparse_prepare_negative_r <= '0;
            sparse_prepare_weight_vector_r <= '0;
            sparse_delta_valid_r <= 1'b0;
            for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                dot_sum_lo_r[pair_idx] <= 20'd0;
                dot_sum_hi_r[pair_idx] <= 20'd0;
                for (int reduce_pair = 0; reduce_pair < 4; reduce_pair++) begin
                    dot_pair_lo_r[pair_idx][reduce_pair] <= 17'd0;
                    dot_pair_hi_r[pair_idx][reduce_pair] <= 17'd0;
                end
                for (int reduce_half = 0; reduce_half < 2; reduce_half++) begin
                    dot_half_lo_r[pair_idx][reduce_half] <= 18'd0;
                    dot_half_hi_r[pair_idx][reduce_half] <= 18'd0;
                end
            end
            for (int pair_idx = 0; pair_idx < 4; pair_idx++) begin
                dot8_row_pair_r[pair_idx] <= 9'd0;
            end
            dot8_row_half_r[0] <= 10'd0;
            dot8_row_half_r[1] <= 10'd0;
            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                accumulator_clear_row_r[row_idx] <= 1'b0;
                pread_accumulator_by_row_r[row_idx] <= '0;
                row_sum[row_idx] <= 32'd0;
                row_increment_r[row_idx] <= 16'd0;
                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                    dense_weight_lo_r[row_idx][pair_idx] <= 8'd0;
                    dense_weight_hi_r[row_idx][pair_idx] <= 8'd0;
                end
                for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                    accumulator[row_idx][column_idx] <= '0;
                    accumulate_delta_r[row_idx][column_idx] <= 21'sd0;
                    sparse_delta_r[row_idx][column_idx] <= 9'sd0;
                end
            end
            for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                column_sum_clear_r[column_idx] <= 1'b0;
                column_sum[column_idx] <= 32'd0;
                column_increment_r[column_idx] <= 16'd0;
                for (int pair_idx = 0; pair_idx < 4; pair_idx++) begin
                    dot8_column_pair_r[column_idx][pair_idx] <= 9'd0;
                end
                dot8_column_half_r[column_idx][0] <= 10'd0;
                dot8_column_half_r[column_idx][1] <= 10'd0;
            end
        end else begin
            if (result_fifo_push) begin
                result_fifo_payload_r[result_fifo_write_pointer_r]
                    <= {p5_column_r, p5_row_r, p5_result_r};
                result_fifo_write_pointer_r <= result_fifo_write_pointer_r + 1'b1;
            end
            if (result_fifo_pop) begin
                result_stage_payload_r
                    <= result_fifo_payload_r[result_fifo_read_pointer_r];
                result_stage_last_r
                    <= result_fifo_read_pointer_r == result_fifo_last_address_r;
                result_stage_valid_r <= 1'b1;
                result_fifo_read_pointer_r <= result_fifo_read_pointer_r + 1'b1;
            end else if (result_stage_pop) begin
                result_stage_valid_r <= 1'b0;
            end
            unique case ({result_fifo_push, result_fifo_pop})
                2'b10: begin
                    result_fifo_count_r <= result_fifo_count_r + 1'b1;
                    result_fifo_has_space_r
                        <= result_fifo_count_r
                            < RESULT_FIFO_COUNT_BITS'(RESULT_FIFO_DEPTH - 1);
                    result_fifo_nonempty_r <= 1'b1;
                end
                2'b01: begin
                    result_fifo_count_r <= result_fifo_count_r - 1'b1;
                    result_fifo_has_space_r <= 1'b1;
                    result_fifo_nonempty_r <= result_fifo_count_r != 1;
                end
                default: begin end
            endcase

            // Unconditional products let Vivado absorb these registers into
            // the DSP48E1 PREG stage. The FSM presents operands one cycle
            // before ST_RESTORE_ADD consumes the registered products.
            zero_restore_low_product_r <=
                zero_restore_reduction_operand_r[15:0] * zero_restore_zp_operand_r;
            zero_restore_high_product_r <=
                zero_restore_reduction_operand_r[31:16] * zero_restore_zp_operand_r;
            // This exact unsigned 8x8 product maps to one DSP48E1 PREG. It
            // replaces four LUT multipliers plus a reconstruction carry chain.
            zero_point_product_r <= activation_zp_r * weight_zp_r;
            // Dedicated copies have one DSP load each, allowing Vivado to use
            // the four BREG input stages instead of routing shared zero points.
            p0_row_zp_low_r <= weight_zp_r;
            p0_row_zp_high_r <= weight_zp_r;
            p0_column_zp_low_r <= activation_zp_r;
            p0_column_zp_high_r <= activation_zp_r;

            // The corresponding valid bit is the architectural qualifier, so
            // these pre-read registers may sample continuously. Free-running
            // them removes a 256-register launch_remaining clock-enable tree;
            // samples outside ST_OUTPUT_PIPELINE are never consumed.
            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                pread_accumulator_by_row_r[row_idx]
                    <= accumulator[row_idx][launch_column_r[3:0]];
            end
            pread_row_sum_r <= row_sum[launch_row_r[2:0]];
            pread_column_sum_r <= column_sum[launch_column_r[3:0]];
            pread_row_r <= launch_row_r;
            pread_column_r <= launch_column_r;

            dense_issue_enable_r <= 1'b0;
            dense_outer_tag0_r <= dense_issue_enable_r
                && mode_r == MODE_DENSE_OUTER;
            dense_outer_tag1_r <= dense_outer_tag0_r;
            dense_outer_tag2_r <= dense_outer_tag1_r;
            sparse_prepare_valid_r <= 1'b0;
            sparse_delta_valid_r <= 1'b0;
            accumulate_delta_valid_r <= 1'b0;
            metric_increment_valid_r <= 1'b0;
            sparse_metric_valid_r <= 1'b0;
            if (metric_increment_valid_r) begin
                if (mode_r == MODE_DENSE_OUTER || mode_r == MODE_DENSE_DOT8) begin
                    debug_dense_macs_r <= debug_dense_macs_r
                        + {11'd0, metric_increment_r};
                end else begin
                    debug_spike_adds_r <= debug_spike_adds_r
                        + {11'd0, metric_increment_r};
                end
            end
            if (sparse_metric_valid_r) begin
                metric_increment_r <= sparse_active_count_r * columns_r;
                metric_increment_valid_r <= 1'b1;
            end
            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                accumulator_clear_row_r[row_idx] <= 1'b0;
            end
            for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                column_sum_clear_r[column_idx] <= 1'b0;
            end
            classify_valid_r <= issue_match_valid_r;
            issue_match_valid_r <= issue_accept;
            // The queue head is stable whenever it is not accepted, so capture
            // its payload unconditionally and use issue_match_valid_r as the
            // only validity qualifier. This removes issue_accept from the CE
            // pins of 1,216 payload registers without changing pipeline latency.
            issue_zero_matches_r <= issue_zero_matches_comb;
            issue_match_last_r <= issue_last;
            issue_match_activation_vector_r <= activation_vector;
            issue_match_outer_weight_vector_r <= outer_weight_vector;
            issue_match_dot_weight_vector_r <= dot_weight_vector;
            if (issue_match_valid_r) begin
                classify_is_zero_r <= &issue_zero_matches_r;
                classify_last_r <= issue_match_last_r;
                classify_activation_vector_r <= issue_match_activation_vector_r;
                classify_outer_weight_vector_r <= issue_match_outer_weight_vector_r;
                classify_dot_weight_vector_r <= issue_match_dot_weight_vector_r;
            end
            dot8_pair_valid_r <= 1'b0;
            dot8_increment_half_valid_r <= 1'b0;
            dot_pair_valid_r <= dense_product_valid && mode_r == MODE_DENSE_DOT8;
            dot_half_valid_r <= dot_pair_valid_r;
            dot_sum_valid_r <= dot_half_valid_r;
            sum_update_valid_r <= 1'b0;
            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                if (accumulator_clear_row_r[row_idx]) begin
                    row_sum[row_idx] <= 32'd0;
                end else if (sum_update_valid_r) begin
                    row_sum[row_idx] <= row_sum[row_idx]
                        + {16'd0, row_increment_r[row_idx]};
                end
            end
            for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                if (column_sum_clear_r[column_idx]) begin
                    column_sum[column_idx] <= 32'd0;
                end else if (sum_update_valid_r) begin
                    column_sum[column_idx] <= column_sum[column_idx]
                        + {16'd0, column_increment_r[column_idx]};
                end
            end
            if (dot8_pair_valid_r) begin
                dot8_increment_half_valid_r <= 1'b1;
                dot8_row_half_r[0] <= dot8_row_pair_r[0] + dot8_row_pair_r[1];
                dot8_row_half_r[1] <= dot8_row_pair_r[2] + dot8_row_pair_r[3];
                for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                    dot8_column_half_r[column_idx][0] <=
                        dot8_column_pair_r[column_idx][0]
                        + dot8_column_pair_r[column_idx][1];
                    dot8_column_half_r[column_idx][1] <=
                        dot8_column_pair_r[column_idx][2]
                        + dot8_column_pair_r[column_idx][3];
                end
            end
            if (dot8_increment_half_valid_r) begin
                row_increment_r[0] <= {6'd0, dot8_row_half_r[0]}
                    + {6'd0, dot8_row_half_r[1]};
                for (int row_idx = 1; row_idx < ROW_LANES; row_idx++) begin
                    row_increment_r[row_idx] <= 16'd0;
                end
                for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                    column_increment_r[column_idx]
                        <= {6'd0, dot8_column_half_r[column_idx][0]}
                            + {6'd0, dot8_column_half_r[column_idx][1]};
                end
                sum_update_valid_r <= 1'b1;
            end
            if (dense_product_valid && mode_r == MODE_DENSE_DOT8) begin
                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                    for (int reduce_pair = 0; reduce_pair < 4; reduce_pair++) begin
                        dot_pair_lo_r[pair_idx][reduce_pair] <=
                            mac_product_lo[2*reduce_pair][pair_idx]
                            + mac_product_lo[2*reduce_pair+1][pair_idx];
                        dot_pair_hi_r[pair_idx][reduce_pair] <=
                            mac_product_hi[2*reduce_pair][pair_idx]
                            + mac_product_hi[2*reduce_pair+1][pair_idx];
                    end
                end
            end
            if (dot_pair_valid_r) begin
                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                    dot_half_lo_r[pair_idx][0] <= dot_pair_lo_r[pair_idx][0]
                        + dot_pair_lo_r[pair_idx][1];
                    dot_half_lo_r[pair_idx][1] <= dot_pair_lo_r[pair_idx][2]
                        + dot_pair_lo_r[pair_idx][3];
                    dot_half_hi_r[pair_idx][0] <= dot_pair_hi_r[pair_idx][0]
                        + dot_pair_hi_r[pair_idx][1];
                    dot_half_hi_r[pair_idx][1] <= dot_pair_hi_r[pair_idx][2]
                        + dot_pair_hi_r[pair_idx][3];
                end
            end
            if (dot_half_valid_r) begin
                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                    dot_sum_lo_r[pair_idx] <= {2'd0, dot_half_lo_r[pair_idx][0]}
                        + {2'd0, dot_half_lo_r[pair_idx][1]};
                    dot_sum_hi_r[pair_idx] <= {2'd0, dot_half_hi_r[pair_idx][0]}
                        + {2'd0, dot_half_hi_r[pair_idx][1]};
                end
            end

            if (sparse_prepare_valid_r) begin
                sparse_delta_valid_r <= 1'b1;
                sparse_active_count_r <= popcount8(sparse_prepare_active_r);
                sparse_metric_valid_r <= 1'b1;
                for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                    for (int column_idx = 0; column_idx < COLUMN_LANES;
                            column_idx++) begin
                        logic signed [7:0] signed_weight;
                        logic signed [8:0] extended_weight;
                        signed_weight = $signed(sparse_prepare_weight_vector_r[
                            column_idx*8 +: 8]);
                        extended_weight = {signed_weight[7], signed_weight};
                        if (sparse_prepare_active_r[row_idx]) begin
                            sparse_delta_r[row_idx][column_idx] <=
                                sparse_prepare_negative_r[row_idx]
                                ? -extended_weight : extended_weight;
                        end else begin
                            sparse_delta_r[row_idx][column_idx] <= 9'sd0;
                        end
                    end
                end
            end

            if (dense_accumulate_valid) begin
                mac_retire_count <= mac_retire_count + 1'b1;
                debug_retired_issues <= debug_retired_issues + 1'b1;
            end
            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                    if (accumulator_clear_row_r[row_idx]) begin
                        accumulator[row_idx][column_idx] <= '0;
                    end else if (accumulate_delta_valid_r) begin
                        accumulator[row_idx][column_idx] <=
                            accumulator[row_idx][column_idx]
                            + {{(ACC_BITS-21){accumulate_delta_r[row_idx][column_idx][20]}},
                               accumulate_delta_r[row_idx][column_idx]};
                    end
                end
            end
            if (dense_outer_tag2_r) begin
                accumulate_delta_valid_r <= 1'b1;
                for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                    for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                        accumulate_delta_r[row_idx][2*pair_idx] <=
                            $signed({5'd0, mac_product_lo[row_idx][pair_idx]});
                        accumulate_delta_r[row_idx][2*pair_idx+1] <=
                            $signed({5'd0, mac_product_hi[row_idx][pair_idx]});
                    end
                end
            end else if (dot_sum_valid_r) begin
                accumulate_delta_valid_r <= 1'b1;
                for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                    for (int pair_idx = 0; pair_idx < COLUMN_PAIRS; pair_idx++) begin
                        accumulate_delta_r[row_idx][2*pair_idx] <=
                            row_idx == 0
                            ? $signed({1'b0, dot_sum_lo_r[pair_idx]}) : 21'sd0;
                        accumulate_delta_r[row_idx][2*pair_idx+1] <=
                            row_idx == 0
                            ? $signed({1'b0, dot_sum_hi_r[pair_idx]}) : 21'sd0;
                    end
                end
            end else if (sparse_delta_valid_r) begin
                accumulate_delta_valid_r <= 1'b1;
                for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                    for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                        accumulate_delta_r[row_idx][column_idx] <=
                            {{12{sparse_delta_r[row_idx][column_idx][8]}},
                              sparse_delta_r[row_idx][column_idx]};
                    end
                end
            end

            unique case (state)
                ST_IDLE: begin
                    if (start) begin
                        mode_r <= compute_mode;
                        rows_r <= active_rows;
                        for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                            active_row_mask_r[row_idx] <= row_idx < active_rows;
                        end
                        columns_r <= active_columns;
                        dense_macs_per_issue_r <= compute_mode == MODE_DENSE_DOT8
                            ? {2'd0, active_columns}
                                * 8'd8
                            : {4'd0, active_rows}
                                * {2'd0, active_columns};
                        activation_zp_r <= activation_zero_point;
                        weight_zp_r <= weight_zero_point;
                        config_error <= active_rows == 0
                            || active_rows > 4'(ROW_LANES)
                            || active_columns == 0
                            || active_columns > 6'(COLUMN_LANES)
                            || (compute_mode == MODE_DENSE_DOT8 && active_rows != 1);
                        effective_reduction <= 32'd0;
                        mac_issue_count <= 12'd0;
                        mac_retire_count <= 12'd0;
                        target_mac_retires <= 12'd0;
                        drain_complete_r <= 1'b0;
                        zero_restore_reduction_operand_r <= 32'd0;
                        zero_restore_zp_operand_r <= 16'd0;
                        zero_point_restore_r <= 48'd0;
                        launch_total_r <= 8'd0;
                        launch_remaining_r <= 8'd0;
                        launch_rows_remaining_r <= 4'd0;
                        launch_columns_remaining_r <= 6'd0;
                        launch_row_r <= 4'd0;
                        launch_column_r <= 6'd0;
                        pread_valid_r <= 1'b0;
                        p0_valid_r <= 1'b0;
                        p1_valid_r <= 1'b0;
                        p2_valid_r <= 1'b0;
                        p3_valid_r <= 1'b0;
                        p4_valid_r <= 1'b0;
                        p5_valid_r <= 1'b0;
                        result_fifo_count_r <= '0;
                        result_fifo_has_space_r <= 1'b1;
                        result_fifo_nonempty_r <= 1'b0;
                        result_fifo_write_pointer_r <= '0;
                        result_fifo_read_pointer_r <= '0;
                        result_fifo_last_address_r <= '0;
                        result_stage_valid_r <= 1'b0;
                        result_stage_payload_r <= '0;
                        result_stage_last_r <= 1'b0;
                        dot_sum_valid_r <= 1'b0;
                        dot_pair_valid_r <= 1'b0;
                        dot_half_valid_r <= 1'b0;
                        dot8_pair_valid_r <= 1'b0;
                        dot8_increment_half_valid_r <= 1'b0;
                        sum_update_valid_r <= 1'b0;
                        input_done_r <= 1'b0;
                        issue_match_valid_r <= 1'b0;
                        issue_zero_matches_r <= 8'd0;
                        issue_match_last_r <= 1'b0;
                        classify_valid_r <= 1'b0;
                        classify_is_zero_r <= 1'b0;
                        classify_last_r <= 1'b0;
                        dense_issue_enable_r <= 1'b0;
                        dense_outer_tag0_r <= 1'b0;
                        dense_outer_tag1_r <= 1'b0;
                        dense_outer_tag2_r <= 1'b0;
                        sparse_prepare_valid_r <= 1'b0;
                        sparse_delta_valid_r <= 1'b0;
                        accumulate_delta_valid_r <= 1'b0;
                        debug_accepted_issues <= 32'd0;
                        debug_skipped_issues <= 32'd0;
                        debug_retired_issues <= 32'd0;
                        debug_dense_macs_r <= 19'd0;
                        debug_spike_adds_r <= 19'd0;
                        metric_increment_r <= 8'd0;
                        metric_increment_valid_r <= 1'b0;
                        sparse_active_count_r <= 4'd0;
                        sparse_metric_valid_r <= 1'b0;
                        for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                            accumulator_clear_row_r[row_idx] <= 1'b1;
                        end
                        for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                            column_sum_clear_r[column_idx] <= 1'b1;
                        end
                        state <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    if (classify_valid_r) begin
                        debug_accepted_issues <= debug_accepted_issues + 1'b1;
                        if (classify_is_zero_r) begin
                            debug_skipped_issues <= debug_skipped_issues + 1'b1;
                        end else if (mode_r == MODE_DENSE_OUTER) begin
                            dense_issue_enable_r <= 1'b1;
                            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                                // Inactive rows are never emitted. Computing
                                // them removes active_rows from every DSP input
                                // mux and is therefore both safe and faster.
                                dense_activation_vector_r[row_idx*8 +: 8]
                                    <= classify_activation_vector_r[row_idx*8 +: 8];
                                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS;
                                        pair_idx++) begin
                                    // Inactive columns are never retired, so
                                    // computing them is harmless. Loading all
                                    // lanes removes active_columns from every
                                    // DSP operand mux and shortens the routed
                                    // data path without changing any result.
                                    dense_weight_lo_r[row_idx][pair_idx]
                                        <= classify_outer_weight_vector_r[
                                            (2*pair_idx)*8 +: 8];
                                    dense_weight_hi_r[row_idx][pair_idx]
                                        <= classify_outer_weight_vector_r[
                                            (2*pair_idx+1)*8 +: 8];
                                end
                            end
                            mac_issue_count <= mac_issue_count + 1'b1;
                            metric_increment_r <= dense_macs_per_issue_r;
                            metric_increment_valid_r <= 1'b1;
                            sum_update_valid_r <= 1'b1;
                            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                                row_increment_r[row_idx]
                                    <= {8'd0, classify_activation_vector_r[row_idx*8 +: 8]};
                            end
                            for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                                column_increment_r[column_idx]
                                    <= {8'd0, classify_outer_weight_vector_r[column_idx*8 +: 8]};
                            end
                        end else if (mode_r == MODE_DENSE_DOT8) begin
                            dense_issue_enable_r <= 1'b1;
                            dense_activation_vector_r <= classify_activation_vector_r;
                            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                                for (int pair_idx = 0; pair_idx < COLUMN_PAIRS;
                                        pair_idx++) begin
                                    dense_weight_lo_r[row_idx][pair_idx]
                                        <= classify_dot_weight_vector_r[
                                            ((2*pair_idx)*ROW_LANES+row_idx)*8 +: 8];
                                    dense_weight_hi_r[row_idx][pair_idx]
                                        <= classify_dot_weight_vector_r[
                                            ((2*pair_idx+1)*ROW_LANES+row_idx)*8 +: 8];
                                end
                            end
                            mac_issue_count <= mac_issue_count + 1'b1;
                            metric_increment_r <= dense_macs_per_issue_r;
                            metric_increment_valid_r <= 1'b1;
                            dot8_pair_valid_r <= 1'b1;
                            for (int pair_idx = 0; pair_idx < 4; pair_idx++) begin
                                dot8_row_pair_r[pair_idx] <=
                                    classify_activation_vector_r[(2*pair_idx)*8 +: 8]
                                    + classify_activation_vector_r[(2*pair_idx+1)*8 +: 8];
                            end
                            for (int column_idx = 0; column_idx < COLUMN_LANES; column_idx++) begin
                                for (int pair_idx = 0; pair_idx < 4; pair_idx++) begin
                                    dot8_column_pair_r[column_idx][pair_idx] <=
                                        classify_dot_weight_vector_r[
                                            (column_idx*8+2*pair_idx)*8 +: 8]
                                        + classify_dot_weight_vector_r[
                                            (column_idx*8+2*pair_idx+1)*8 +: 8];
                                end
                            end
                        end else begin
                            sparse_prepare_valid_r <= 1'b1;
                            sparse_prepare_weight_vector_r
                                <= classify_outer_weight_vector_r;
                            for (int row_idx = 0; row_idx < ROW_LANES; row_idx++) begin
                                logic signed [7:0] spike_value;
                                spike_value = $signed(
                                    classify_activation_vector_r[row_idx*8 +: 8]);
                                sparse_prepare_active_r[row_idx]
                                    <= active_row_mask_r[row_idx] && spike_value != 0;
                                sparse_prepare_negative_r[row_idx]
                                    <= mode_r == MODE_TERNARY_WEIGHTED
                                        && spike_value < 0;
                            end
                            debug_retired_issues <= debug_retired_issues + 1'b1;
                        end

                        if (classify_last_r) begin
                            launch_row_r <= 4'd0;
                            launch_column_r <= 6'd0;
                            pread_valid_r <= 1'b0;
                            p0_valid_r <= 1'b0;
                            p1_valid_r <= 1'b0;
                            p2_valid_r <= 1'b0;
                            p3_valid_r <= 1'b0;
                            p4_valid_r <= 1'b0;
                            p5_valid_r <= 1'b0;
                            if (mode_r == MODE_DENSE_OUTER || mode_r == MODE_DENSE_DOT8) begin
                                state <= ST_CAPTURE_TARGET;
                            end else begin
                                state <= ST_SPARSE_DRAIN;
                            end
                        end
                    end

                    if (issue_accept) begin
                        if (issue_last) input_done_r <= 1'b1;
                    end
                end

                ST_DRAIN: begin
                    // End the 12-bit equality at one local register before it
                    // controls the physically replicated state network.
                    drain_complete_r <= mac_retire_count == target_mac_retires;
                    if (drain_complete_r) begin
                        state <= ST_PREPARE_ZERO_POINT;
                    end
                end

                ST_CAPTURE_TARGET: begin
                    target_mac_retires <= mac_issue_count;
                    effective_reduction <= mode_r == MODE_DENSE_DOT8
                        ? {17'd0, mac_issue_count, 3'd0}
                        : {20'd0, mac_issue_count};
                    drain_complete_r <= 1'b0;
                    state <= ST_DRAIN;
                end

                ST_SPARSE_DRAIN: begin
                    state <= ST_PREPARE_OUTPUT_COUNT;
                end

                ST_PREPARE_ZERO_POINT: begin
                    state <= ST_LATCH_ZERO_POINT;
                end

                ST_LATCH_ZERO_POINT: begin
                    state <= ST_COMBINE_ZERO_POINT;
                end

                ST_COMBINE_ZERO_POINT: begin
                    zero_restore_reduction_operand_r <= effective_reduction;
                    zero_restore_zp_operand_r <= zero_point_product_r;
                    state <= ST_RESTORE_MULTIPLY;
                end

                ST_RESTORE_MULTIPLY: begin
                    state <= ST_RESTORE_RELAY;
                end

                ST_RESTORE_RELAY: begin
                    // Isolate the two multiplier PREGs from the reconstruction
                    // adder. Vivado can absorb these registers into the adder's
                    // DSP input stage, replacing one DSP-to-DSP combinational
                    // timing path with two independently timed pipeline paths.
                    zero_restore_low_relay_r <= zero_restore_low_product_r;
                    zero_restore_high_relay_r <= zero_restore_high_product_r;
                    state <= ST_RESTORE_ADD;
                end

                ST_RESTORE_ADD: begin
                    zero_point_restore_r <= {16'd0, zero_restore_low_relay_r}
                        + {zero_restore_high_relay_r, 16'd0};
                    state <= ST_PREPARE_OUTPUT_COUNT;
                end

                ST_PREPARE_OUTPUT_COUNT: begin
                    launch_total_r <= {2'd0, columns_r};
                    launch_rows_remaining_r <= rows_r;
                    launch_columns_remaining_r <= columns_r;
                    state <= ST_ACCUMULATE_OUTPUT_COUNT;
                end

                ST_ACCUMULATE_OUTPUT_COUNT: begin
                    if (launch_rows_remaining_r <= 4'd1) begin
                        launch_remaining_r <= launch_total_r;
                        // Output addresses start at zero. Truncation is exact
                        // for the 128-result case: 128 mod 128 minus one is 127.
                        result_fifo_last_address_r
                            <= launch_total_r[RESULT_FIFO_ADDRESS_BITS-1:0] - 1'b1;
                        state <= ST_OUTPUT_PIPELINE;
                    end else begin
                        launch_total_r <= launch_total_r
                            + {2'd0, columns_r};
                        launch_rows_remaining_r <= launch_rows_remaining_r - 1'b1;
                    end
                end

                ST_OUTPUT_PIPELINE: begin
                    if (output_pipeline_enable) begin
                        p5_valid_r <= p4_valid_r;
                        if (p4_valid_r) begin
                            p5_result_r <= 48'(p4_corrected_r
                                + $signed({16'd0, zero_point_restore_r}));
                            p5_row_r <= p4_row_r;
                            p5_column_r <= p4_column_r;
                        end

                        p4_valid_r <= p3_valid_r;
                        if (p3_valid_r) begin
                            p4_corrected_r <= p3_corrected_r
                                - p3_column_correction_r;
                            p4_row_r <= p3_row_r;
                            p4_column_r <= p3_column_r;
                        end

                        p3_valid_r <= p2_valid_r;
                        if (p2_valid_r) begin
                            p3_corrected_r <= p2_raw_r - p2_row_correction_r;
                            p3_column_correction_r <= p2_column_correction_r;
                            p3_row_r <= p2_row_r;
                            p3_column_r <= p2_column_r;
                        end

                        p2_valid_r <= p1_valid_r;
                        if (p1_valid_r) begin
                            p2_raw_r <= p1_raw_r;
                            p2_row_correction_r <= $signed({40'd0,
                                p1_row_product_low_r})
                                + $signed({24'd0, p1_row_product_high_r, 16'd0});
                            p2_column_correction_r <= $signed({40'd0,
                                p1_column_product_low_r})
                                + $signed({24'd0, p1_column_product_high_r, 16'd0});
                            p2_row_r <= p1_row_r;
                            p2_column_r <= p1_column_r;
                        end

                        p1_valid_r <= p0_valid_r;
                        if (p0_valid_r) begin
                            p1_raw_r <= p0_raw_r;
                            p1_row_product_low_r <= p0_row_product_low;
                            p1_row_product_high_r <= p0_row_product_high;
                            p1_column_product_low_r <= p0_column_product_low;
                            p1_column_product_high_r <= p0_column_product_high;
                            p1_row_r <= p0_row_r;
                            p1_column_r <= p0_column_r;
                        end

                        // Split the 128-entry accumulator read into a 16-column
                        // selection and an 8-row selection. This adds one fill
                        // cycle while preserving one result per clock.
                        p0_valid_r <= pread_valid_r;
                        if (pread_valid_r) begin
                            p0_raw_r <= {{(64-ACC_BITS){
                                pread_accumulator_by_row_r[pread_row_r[2:0]][ACC_BITS-1]}},
                                pread_accumulator_by_row_r[pread_row_r[2:0]]};
                            p0_row_sum_r <= pread_row_sum_r;
                            p0_column_sum_r <= pread_column_sum_r;
                            p0_row_r <= pread_row_r;
                            p0_column_r <= pread_column_r;
                        end

                        pread_valid_r <= launch_remaining_r != 0;
                        if (launch_remaining_r != 0) begin
                            launch_remaining_r <= launch_remaining_r - 1'b1;
                            launch_row_r <= launch_row_r
                                + (launch_columns_remaining_r == 6'd1);
                            if (launch_columns_remaining_r == 6'd1) begin
                                launch_columns_remaining_r <= columns_r;
                                launch_column_r <= 6'd0;
                            end else begin
                                launch_columns_remaining_r <=
                                    launch_columns_remaining_r - 1'b1;
                                launch_column_r <= launch_column_r + 1'b1;
                            end
                        end else begin
                            pread_valid_r <= 1'b0;
                        end
                    end
                    if (result_stage_pop && result_stage_last_r) state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (ROW_LANES != 8) $fatal(1, "neuraedge_tensor_tile currently requires ROW_LANES=8");
        if (COLUMN_LANES != 16) $fatal(1, "neuraedge_tensor_tile currently requires COLUMN_LANES=16");
        if (COLUMN_LANES % 2 != 0) $fatal(1, "COLUMN_LANES must be even");
        if (ACC_BITS < 32 || ACC_BITS > 48) $fatal(1, "ACC_BITS must be in [32,48]");
    end
endmodule

`default_nettype wire
