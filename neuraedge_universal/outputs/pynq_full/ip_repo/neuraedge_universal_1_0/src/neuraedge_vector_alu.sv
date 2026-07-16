`timescale 1ns / 1ps
`default_nettype none

module neuraedge_vector_alu #(
    parameter int LANES = 8,
    parameter int DATA_BITS = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire input_valid,
    output logic input_ready,
    input  wire [3:0] operation,
    input  wire [LANES*DATA_BITS-1:0] operand_a,
    input  wire [LANES*DATA_BITS-1:0] operand_b,
    input  wire signed [15:0] multiplier_q15,
    input  wire [4:0] right_shift,
    input  wire signed [15:0] output_zero_point,
    input  wire signed [DATA_BITS-1:0] threshold,
    input  wire [DATA_BITS-1:0] surrogate_window,
    input  wire signed [DATA_BITS-1:0] clamp_minimum,
    input  wire signed [DATA_BITS-1:0] clamp_maximum,
    output logic output_valid,
    input  wire output_ready,
    output logic [LANES*DATA_BITS-1:0] result,
    output logic config_error
);
    localparam logic [3:0] OP_ADD_SAT = 4'd0;
    localparam logic [3:0] OP_RELU = 4'd1;
    localparam logic [3:0] OP_REQUANTIZE = 4'd2;
    localparam logic [3:0] OP_BINARY_SPIKE = 4'd3;
    localparam logic [3:0] OP_TERNARY_SPIKE = 4'd4;
    localparam logic [3:0] OP_SURROGATE_GRADIENT = 4'd5;
    localparam logic [3:0] OP_MAXIMUM = 4'd6;
    localparam logic [3:0] OP_MULTIPLY_Q15 = 4'd7;
    localparam logic [3:0] OP_CLAMP = 4'd8;
    localparam logic [3:0] OP_FLEX_REQUANTIZE = 4'd9;
    localparam logic [3:0] OP_SELECT_ZERO = 4'd10;
    localparam logic [3:0] OP_SELECT_NONZERO = 4'd11;
    localparam logic [3:0] OP_COMPARE_GE = 4'd12;

    // One arithmetic transformation is performed per state.  Besides making the
    // latency deterministic, these registers prevent a multiply, absolute value,
    // rounding add, barrel shift, sign restore, and clamp from becoming one path.
    typedef enum logic [4:0] {
        ST_IDLE = 5'd0,
        ST_LOAD_LANE = 5'd1,
        ST_MULTIPLY = 5'd2,
        ST_ABSOLUTE = 5'd3,
        ST_ROUND_BIAS = 5'd4,
        ST_SHIFT = 5'd5,
        ST_RESTORE_SIGN = 5'd6,
        ST_OFFSET = 5'd7,
        ST_PRECOMPUTE = 5'd8,
        ST_COMPARE = 5'd9,
        ST_RESULT = 5'd10,
        ST_OUTPUT = 5'd11,
        ST_MULTIPLIER_LATCH = 5'd12,
        ST_PRODUCT_LATCH = 5'd13,
        ST_PRODUCT_CROSS = 5'd14,
        ST_PRODUCT_LOW = 5'd15,
        ST_OFFSET_ADD = 5'd16,
        ST_OFFSET_EXTEND = 5'd17,
        ST_DISTANCE_ABSOLUTE = 5'd18,
        ST_STORE_RESULT = 5'd19
    } state_t;

    state_t state;
    logic [2:0] lane_index;
    logic [3:0] operation_r;
    logic [LANES*DATA_BITS-1:0] operand_a_r;
    logic [LANES*DATA_BITS-1:0] operand_b_r;
    logic signed [15:0] multiplier_q15_r;
    logic signed [15:0] output_zero_point_r;
    logic signed [31:0] threshold_r;
    logic signed [31:0] negative_threshold_r;
    logic [31:0] surrogate_window_r;
    logic signed [31:0] clamp_minimum_r;
    logic signed [31:0] clamp_maximum_r;
    logic [5:0] shift_amount_r;
    logic [63:0] rounding_bias_r;

    logic signed [31:0] lane_a_r;
    logic signed [31:0] lane_b_r;
    logic signed [31:0] multiplier_left_r;
    logic signed [31:0] multiplier_right_r;
    logic signed [16:0] multiplier_left_low_r;
    logic signed [16:0] multiplier_right_low_r;
    logic signed [15:0] multiplier_left_high_r;
    logic signed [15:0] multiplier_right_high_r;
    (* use_dsp = "yes" *) wire signed [33:0] multiplier_product_ll =
        multiplier_left_low_r * multiplier_right_low_r;
    (* use_dsp = "yes" *) wire signed [32:0] multiplier_product_lh =
        multiplier_left_low_r * multiplier_right_high_r;
    (* use_dsp = "yes" *) wire signed [32:0] multiplier_product_hl =
        multiplier_left_high_r * multiplier_right_low_r;
    (* use_dsp = "yes" *) wire signed [31:0] multiplier_product_hh =
        multiplier_left_high_r * multiplier_right_high_r;
    logic signed [33:0] product_ll_r;
    logic signed [32:0] product_lh_r;
    logic signed [32:0] product_hl_r;
    logic signed [31:0] product_hh_r;
    logic signed [33:0] product_cross_r;
    logic signed [63:0] product_low_r;
    logic signed [63:0] product_latched_r;
    logic product_negative_r;
    logic [63:0] magnitude_r;
    logic [63:0] biased_magnitude_r;
    logic [63:0] shifted_magnitude_r;
    logic signed [63:0] rounded_signed_r;
    logic signed [32:0] offset_left_r;
    logic signed [32:0] offset_right_r;
    logic signed [32:0] offset_sum_r;
    logic signed [63:0] wide_result_r;
    logic signed [31:0] saturated_value_r;
    logic [31:0] distance_r;
    logic signed [32:0] distance_difference_r;
    logic lane_a_negative_r;
    logic lane_a_ge_threshold_r;
    logic lane_a_le_negative_threshold_r;
    logic lane_a_ge_b_r;
    logic lane_a_zero_r;
    logic lane_a_below_clamp_r;
    logic lane_a_above_clamp_r;
    logic saturated_below_clamp_r;
    logic saturated_above_clamp_r;
    logic inside_surrogate_window_r;
    logic signed [31:0] lane_result_r;

    function automatic logic signed [31:0] saturate_s32(
        input logic signed [63:0] value
    );
        begin
            // A signed 64-bit number fits in 32 bits exactly when every upper bit
            // repeats bit 31.  This avoids a long pair of 64-bit comparators.
            if (value[63:32] == {32{value[31]}}) saturate_s32 = value[31:0];
            else if (value[63]) saturate_s32 = 32'sh80000000;
            else saturate_s32 = 32'sh7fffffff;
        end
    endfunction

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            lane_index <= 3'd0;
            operation_r <= OP_ADD_SAT;
            operand_a_r <= '0;
            operand_b_r <= '0;
            multiplier_q15_r <= 16'sd0;
            output_zero_point_r <= 16'sd0;
            threshold_r <= 32'sd0;
            negative_threshold_r <= 32'sd0;
            surrogate_window_r <= 32'd0;
            clamp_minimum_r <= 32'sh80000000;
            clamp_maximum_r <= 32'sh7fffffff;
            shift_amount_r <= 6'd0;
            rounding_bias_r <= 64'd0;
            lane_a_r <= 32'sd0;
            lane_b_r <= 32'sd0;
            multiplier_left_r <= 32'sd0;
            multiplier_right_r <= 32'sd0;
            multiplier_left_low_r <= 17'sd0;
            multiplier_right_low_r <= 17'sd0;
            multiplier_left_high_r <= 16'sd0;
            multiplier_right_high_r <= 16'sd0;
            product_ll_r <= 34'sd0;
            product_lh_r <= 33'sd0;
            product_hl_r <= 33'sd0;
            product_hh_r <= 32'sd0;
            product_cross_r <= 34'sd0;
            product_low_r <= 64'sd0;
            product_latched_r <= 64'sd0;
            product_negative_r <= 1'b0;
            magnitude_r <= 64'd0;
            biased_magnitude_r <= 64'd0;
            shifted_magnitude_r <= 64'd0;
            rounded_signed_r <= 64'sd0;
            offset_left_r <= 33'sd0;
            offset_right_r <= 33'sd0;
            offset_sum_r <= 33'sd0;
            wide_result_r <= 64'sd0;
            saturated_value_r <= 32'sd0;
            distance_r <= 32'd0;
            distance_difference_r <= 33'sd0;
            lane_a_negative_r <= 1'b0;
            lane_a_ge_threshold_r <= 1'b0;
            lane_a_le_negative_threshold_r <= 1'b0;
            lane_a_ge_b_r <= 1'b0;
            lane_a_zero_r <= 1'b0;
            lane_a_below_clamp_r <= 1'b0;
            lane_a_above_clamp_r <= 1'b0;
            saturated_below_clamp_r <= 1'b0;
            saturated_above_clamp_r <= 1'b0;
            inside_surrogate_window_r <= 1'b0;
            lane_result_r <= 32'sd0;
            result <= '0;
            config_error <= 1'b0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        operation_r <= operation;
                        operand_a_r <= operand_a;
                        operand_b_r <= operand_b;
                        multiplier_q15_r <= multiplier_q15;
                        output_zero_point_r <= output_zero_point;
                        threshold_r <= threshold;
                        negative_threshold_r <= -threshold;
                        surrogate_window_r <= surrogate_window;
                        clamp_minimum_r <= clamp_minimum;
                        clamp_maximum_r <= clamp_maximum;
                        config_error <= operation > OP_COMPARE_GE
                            || clamp_minimum > clamp_maximum
                            || (operation == OP_FLEX_REQUANTIZE
                                && right_shift == 5'b10000);
                        lane_index <= 3'd0;
                        if (operation == OP_REQUANTIZE) begin
                            shift_amount_r <= 6'd15 + {1'b0, right_shift};
                            rounding_bias_r <= 64'd16384 << right_shift;
                        end else if (operation == OP_FLEX_REQUANTIZE) begin
                            logic signed [5:0] effective_shift;
                            effective_shift = 6'sd15
                                + $signed({right_shift[4], right_shift});
                            if (right_shift == 5'b10000) begin
                                shift_amount_r <= 6'd0;
                                rounding_bias_r <= 64'd0;
                            end else begin
                                shift_amount_r <= effective_shift[5:0];
                                rounding_bias_r <= effective_shift == 0
                                    ? 64'd0
                                    : (64'd1 << (effective_shift - 1'b1));
                            end
                        end else if (operation == OP_MULTIPLY_Q15) begin
                            shift_amount_r <= 6'd15;
                            rounding_bias_r <= 64'd16384;
                        end else begin
                            shift_amount_r <= 6'd0;
                            rounding_bias_r <= 64'd0;
                        end
                        state <= ST_LOAD_LANE;
                    end
                end

                ST_LOAD_LANE: begin
                    lane_a_r <= $signed(
                        operand_a_r[lane_index*DATA_BITS +: DATA_BITS]);
                    lane_b_r <= $signed(
                        operand_b_r[lane_index*DATA_BITS +: DATA_BITS]);
                    state <= ST_MULTIPLIER_LATCH;
                end

                ST_MULTIPLIER_LATCH: begin
                    multiplier_left_r <= lane_a_r;
                    if (operation_r == OP_REQUANTIZE
                            || operation_r == OP_FLEX_REQUANTIZE) begin
                        multiplier_right_r <= {
                            {16{multiplier_q15_r[15]}}, multiplier_q15_r};
                    end else if (operation_r == OP_MULTIPLY_Q15) begin
                        multiplier_right_r <= lane_b_r;
                    end else begin
                        multiplier_right_r <= 32'sd0;
                    end
                    state <= ST_MULTIPLY;
                end

                ST_MULTIPLY: begin
                    multiplier_left_low_r <= $signed({1'b0, multiplier_left_r[15:0]});
                    multiplier_right_low_r <= $signed({1'b0, multiplier_right_r[15:0]});
                    multiplier_left_high_r <= multiplier_left_r[31:16];
                    multiplier_right_high_r <= multiplier_right_r[31:16];
                    state <= ST_PRODUCT_CROSS;
                end

                ST_PRODUCT_CROSS: begin
                    product_ll_r <= multiplier_product_ll;
                    product_lh_r <= multiplier_product_lh;
                    product_hl_r <= multiplier_product_hl;
                    product_hh_r <= multiplier_product_hh;
                    state <= ST_PRODUCT_LOW;
                end

                ST_PRODUCT_LOW: begin
                    product_cross_r <= $signed({product_lh_r[32], product_lh_r})
                        + $signed({product_hl_r[32], product_hl_r});
                    product_low_r <= $signed({{30{product_ll_r[33]}}, product_ll_r});
                    state <= ST_PRODUCT_LATCH;
                end

                ST_PRODUCT_LATCH: begin
                    product_latched_r <= product_low_r
                        + ($signed({{30{product_cross_r[33]}}, product_cross_r}) <<< 16)
                        + ($signed({{32{product_hh_r[31]}}, product_hh_r}) <<< 32);
                    state <= ST_ABSOLUTE;
                end

                ST_ABSOLUTE: begin
                    product_negative_r <= product_latched_r[63];
                    magnitude_r <= product_latched_r[63]
                        ? -product_latched_r : product_latched_r;
                    state <= ST_ROUND_BIAS;
                end

                ST_ROUND_BIAS: begin
                    biased_magnitude_r <= magnitude_r + rounding_bias_r;
                    state <= ST_SHIFT;
                end

                ST_SHIFT: begin
                    shifted_magnitude_r <= biased_magnitude_r >> shift_amount_r;
                    state <= ST_RESTORE_SIGN;
                end

                ST_RESTORE_SIGN: begin
                    rounded_signed_r <= product_negative_r
                        ? -$signed(shifted_magnitude_r)
                        : $signed(shifted_magnitude_r);
                    state <= ST_OFFSET;
                end

                ST_OFFSET: begin
                    unique case (operation_r)
                        OP_ADD_SAT: begin
                            offset_left_r <= {lane_a_r[31], lane_a_r};
                            offset_right_r <= {lane_b_r[31], lane_b_r};
                            state <= ST_OFFSET_ADD;
                        end
                        OP_REQUANTIZE,
                        OP_FLEX_REQUANTIZE: begin
                            offset_left_r <= rounded_signed_r[32:0];
                            offset_right_r <= {{17{output_zero_point_r[15]}},
                                               output_zero_point_r};
                            state <= ST_OFFSET_ADD;
                        end
                        OP_MULTIPLY_Q15: begin
                            wide_result_r <= rounded_signed_r;
                            state <= ST_PRECOMPUTE;
                        end
                        default: begin
                            wide_result_r <= {{32{lane_a_r[31]}}, lane_a_r};
                            state <= ST_PRECOMPUTE;
                        end
                    endcase
                end

                ST_OFFSET_ADD: begin
                    offset_sum_r <= offset_left_r + offset_right_r;
                    state <= ST_OFFSET_EXTEND;
                end

                ST_OFFSET_EXTEND: begin
                    wide_result_r <= {{31{offset_sum_r[32]}}, offset_sum_r};
                    state <= ST_PRECOMPUTE;
                end

                ST_PRECOMPUTE: begin
                    saturated_value_r <= saturate_s32(wide_result_r);
                    distance_difference_r <= {lane_a_r[31], lane_a_r}
                        - {threshold_r[31], threshold_r};
                    lane_a_negative_r <= lane_a_r < 0;
                    lane_a_ge_threshold_r <= lane_a_r >= threshold_r;
                    lane_a_le_negative_threshold_r <= lane_a_r <= negative_threshold_r;
                    lane_a_ge_b_r <= lane_a_r >= lane_b_r;
                    lane_a_zero_r <= lane_a_r == 0;
                    state <= ST_DISTANCE_ABSOLUTE;
                end

                ST_DISTANCE_ABSOLUTE: begin
                    distance_r <= distance_difference_r[32]
                        ? (~distance_difference_r[31:0] + 1'b1)
                        : distance_difference_r[31:0];
                    state <= ST_COMPARE;
                end

                ST_COMPARE: begin
                    inside_surrogate_window_r <= distance_r <= surrogate_window_r;
                    lane_a_below_clamp_r <= lane_a_r < clamp_minimum_r;
                    lane_a_above_clamp_r <= lane_a_r > clamp_maximum_r;
                    saturated_below_clamp_r <= saturated_value_r < clamp_minimum_r;
                    saturated_above_clamp_r <= saturated_value_r > clamp_maximum_r;
                    state <= ST_RESULT;
                end

                ST_RESULT: begin
                    logic signed [31:0] lane_result;
                    lane_result = 32'sd0;

                    unique case (operation_r)
                        OP_ADD_SAT: lane_result = saturated_value_r;
                        OP_RELU: lane_result = lane_a_negative_r ? 32'sd0 : lane_a_r;
                        OP_REQUANTIZE,
                        OP_FLEX_REQUANTIZE: begin
                            if (saturated_below_clamp_r) begin
                                lane_result = clamp_minimum_r;
                            end else if (saturated_above_clamp_r) begin
                                lane_result = clamp_maximum_r;
                            end else begin
                                lane_result = saturated_value_r;
                            end
                        end
                        OP_BINARY_SPIKE: lane_result = lane_a_ge_threshold_r
                            ? 32'sd1 : 32'sd0;
                        OP_TERNARY_SPIKE: begin
                            if (lane_a_ge_threshold_r) lane_result = 32'sd1;
                            else if (lane_a_le_negative_threshold_r) lane_result = -32'sd1;
                            else lane_result = 32'sd0;
                        end
                        OP_SURROGATE_GRADIENT: lane_result =
                            inside_surrogate_window_r ? lane_b_r : 32'sd0;
                        OP_MAXIMUM: lane_result = lane_a_ge_b_r
                            ? lane_a_r : lane_b_r;
                        OP_MULTIPLY_Q15: lane_result = saturated_value_r;
                        OP_CLAMP: begin
                            if (lane_a_below_clamp_r) lane_result = clamp_minimum_r;
                            else if (lane_a_above_clamp_r) lane_result = clamp_maximum_r;
                            else lane_result = lane_a_r;
                        end
                        OP_SELECT_ZERO: lane_result = lane_a_zero_r
                            ? lane_b_r : 32'sd0;
                        OP_SELECT_NONZERO: lane_result = lane_a_zero_r
                            ? 32'sd0 : lane_b_r;
                        OP_COMPARE_GE: lane_result = lane_a_ge_b_r
                            ? 32'sd1 : 32'sd0;
                        default: lane_result = 32'sd0;
                    endcase
                    // Keep the operation-select mux local to the scalar lane
                    // pipeline. The following state moves this registered value
                    // into the wide retained output vector, avoiding a routed
                    // comparison-to-distant-result path across the device.
                    lane_result_r <= lane_result;
                    state <= ST_STORE_RESULT;
                end

                ST_STORE_RESULT: begin
                    result[lane_index*DATA_BITS +: DATA_BITS] <= lane_result_r;
                    if (lane_index == $clog2(LANES)'(LANES - 1)) begin
                        state <= ST_OUTPUT;
                    end else begin
                        lane_index <= lane_index + 1'b1;
                        state <= ST_LOAD_LANE;
                    end
                end

                ST_OUTPUT: if (output_ready) state <= ST_IDLE;
                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (LANES != 8) $fatal(1, "neuraedge_vector_alu currently requires LANES=8");
        if (DATA_BITS != 32) $fatal(1, "neuraedge_vector_alu currently requires DATA_BITS=32");
    end
endmodule

`default_nettype wire
