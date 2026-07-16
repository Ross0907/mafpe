`timescale 1ns / 1ps
`default_nettype none

module neuraedge_log_softmax #(
    parameter int MAX_ITEMS = 256,
    parameter int SCORE_BITS = 24
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start_valid,
    output logic start_ready,
    input  wire [8:0] item_count,
    input  wire [4:0] difference_shift,
    input  wire score_valid,
    output logic score_ready,
    input  wire signed [SCORE_BITS-1:0] score_value,
    input  wire score_last,
    output logic probability_valid,
    input  wire probability_ready,
    output logic [15:0] probability_q15,
    output logic [8:0] probability_index,
    output logic probability_last,
    output logic config_error,
    output logic [31:0] cycle_count,
    output logic signed [SCORE_BITS-1:0] debug_maximum_score,
    output logic [23:0] debug_denominator,
    output logic [31:0] debug_reciprocal_q31
);
    typedef enum logic [4:0] {
        ST_IDLE = 5'd0,
        ST_LOAD = 5'd1,
        ST_EXPONENT_READ = 5'd2,
        ST_EXPONENT_DIFFERENCE = 5'd3,
        ST_EXPONENT_QUANTIZE = 5'd4,
        ST_EXPONENT_ACCUMULATE = 5'd5,
        ST_RECIPROCAL_START = 5'd6,
        ST_RECIPROCAL_WAIT = 5'd7,
        ST_OUTPUT_READ = 5'd8,
        ST_OUTPUT_MULTIPLY = 5'd9,
        ST_OUTPUT_ROUND = 5'd10,
        ST_OUTPUT_HOLD = 5'd11,
        ST_EXPONENT_LATCH = 5'd12,
        ST_OUTPUT_LATCH = 5'd13,
        ST_OUTPUT_OPERANDS = 5'd14,
        ST_OUTPUT_PRODUCT_ADD = 5'd15,
        ST_EXPONENT_SHIFT = 5'd16,
        ST_EXPONENT_CLAMP = 5'd17
    } state_t;

    (* fsm_encoding = "one_hot" *) state_t state;
    (* ram_style = "block" *) logic signed [SCORE_BITS-1:0] score_mem [0:MAX_ITEMS-1];
    (* ram_style = "block" *) logic [15:0] exponent_mem [0:MAX_ITEMS-1];
    logic [8:0] item_count_r;
    logic [4:0] difference_shift_r;
    logic [8:0] load_index;
    logic [8:0] load_remaining_r;
    logic [8:0] exponent_index;
    logic [8:0] exponent_remaining_r;
    logic [8:0] output_index_r;
    logic [8:0] output_remaining_r;
    logic signed [SCORE_BITS-1:0] maximum_score_r;
    logic [23:0] denominator_r;
    logic [31:0] reciprocal_q31_r;
    logic [15:0] output_exponent_r;
    logic [15:0] output_exponent_ram_r;
    logic [15:0] multiply_exponent_r;
    logic [31:0] multiply_reciprocal_r;
    logic signed [SCORE_BITS-1:0] exponent_score_r;
    logic signed [SCORE_BITS-1:0] exponent_score_ram_r;
    logic [SCORE_BITS:0] exponent_difference_r;
    logic [SCORE_BITS:0] shifted_difference_r;
    logic [4:0] exponent_r;
    logic [15:0] unnormalized_r;
    logic [31:0] probability_product_r;
    logic [31:0] probability_low_product_r;
    logic [31:0] probability_high_product_r;
    logic probability_last_r;
    (* use_dsp = "yes" *) wire [31:0] probability_low_product =
        multiply_exponent_r * multiply_reciprocal_r[15:0];
    (* use_dsp = "yes" *) wire [31:0] probability_high_product =
        multiply_exponent_r * multiply_reciprocal_r[31:16];

    logic divider_input_valid;
    wire divider_input_ready;
    wire divider_output_valid;
    logic divider_output_ready;
    wire [31:0] divider_quotient;
    wire divider_zero;

    neuraedge_udivider #(.NUMERATOR_BITS(32)) u_divider (
        .clk(clk), .rst_n(rst_n),
        .input_valid(divider_input_valid), .input_ready(divider_input_ready),
        .numerator(32'd2147483647), .denominator({8'd0, denominator_r}),
        .output_valid(divider_output_valid), .output_ready(divider_output_ready),
        .quotient(divider_quotient), .remainder(),
        .divide_by_zero(divider_zero)
    );

    always_comb begin
        start_ready = state == ST_IDLE;
        score_ready = state == ST_LOAD;
        probability_valid = state == ST_OUTPUT_HOLD;
        probability_index = output_index_r;
        probability_last = probability_last_r;
        divider_input_valid = state == ST_RECIPROCAL_START;
        divider_output_ready = state == ST_RECIPROCAL_WAIT;
        debug_maximum_score = maximum_score_r;
        debug_denominator = denominator_r;
        debug_reciprocal_q31 = reciprocal_q31_r;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            item_count_r <= 9'd0;
            difference_shift_r <= 5'd0;
            load_index <= 9'd0;
            load_remaining_r <= 9'd0;
            exponent_index <= 9'd0;
            exponent_remaining_r <= 9'd0;
            output_index_r <= 9'd0;
            output_remaining_r <= 9'd0;
            maximum_score_r <= {1'b1, {(SCORE_BITS-1){1'b0}}};
            denominator_r <= 24'd0;
            reciprocal_q31_r <= 32'd0;
            output_exponent_r <= 16'd0;
            output_exponent_ram_r <= 16'd0;
            multiply_exponent_r <= 16'd0;
            multiply_reciprocal_r <= 32'd0;
            exponent_score_r <= '0;
            exponent_score_ram_r <= '0;
            exponent_difference_r <= '0;
            shifted_difference_r <= '0;
            exponent_r <= 5'd0;
            unnormalized_r <= 16'd0;
            probability_product_r <= 32'd0;
            probability_low_product_r <= 32'd0;
            probability_high_product_r <= 32'd0;
            probability_last_r <= 1'b0;
            probability_q15 <= 16'd0;
            config_error <= 1'b0;
            cycle_count <= 32'd0;
        end else begin
            if (state != ST_IDLE) cycle_count <= cycle_count + 1'b1;
            unique case (state)
                ST_IDLE: begin
                    if (start_valid) begin
                        cycle_count <= 32'd0;
                        config_error <= item_count == 0
                            || item_count > 9'(MAX_ITEMS);
                        if (item_count == 0 || item_count > 9'(MAX_ITEMS)) begin
                            state <= ST_IDLE;
                        end else begin
                            item_count_r <= item_count;
                            difference_shift_r <= difference_shift;
                            load_index <= 9'd0;
                            load_remaining_r <= item_count;
                            maximum_score_r <= {1'b1, {(SCORE_BITS-1){1'b0}}};
                            denominator_r <= 24'd0;
                            state <= ST_LOAD;
                        end
                    end
                end

                ST_LOAD: begin
                    if (score_valid) begin
                        logic expected_last;
                        expected_last = load_remaining_r == 9'd1;
                        score_mem[load_index[7:0]] <= score_value;
                        if (score_value > maximum_score_r) maximum_score_r <= score_value;
                        if (score_last != expected_last) config_error <= 1'b1;
                        if (expected_last) begin
                            exponent_index <= 9'd0;
                            exponent_remaining_r <= item_count_r;
                            state <= ST_EXPONENT_READ;
                        end else if (score_last) begin
                            config_error <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            load_index <= load_index + 1'b1;
                            load_remaining_r <= load_remaining_r - 1'b1;
                        end
                    end
                end

                ST_EXPONENT_READ: begin
                    exponent_score_ram_r <= score_mem[exponent_index[7:0]];
                    state <= ST_EXPONENT_LATCH;
                end

                ST_EXPONENT_LATCH: begin
                    exponent_score_r <= exponent_score_ram_r;
                    state <= ST_EXPONENT_DIFFERENCE;
                end

                ST_EXPONENT_DIFFERENCE: begin
                    exponent_difference_r <=
                        $signed(maximum_score_r) - $signed(exponent_score_r);
                    state <= ST_EXPONENT_SHIFT;
                end

                ST_EXPONENT_SHIFT: begin
                    shifted_difference_r <= exponent_difference_r >> difference_shift_r;
                    state <= ST_EXPONENT_CLAMP;
                end

                ST_EXPONENT_CLAMP: begin
                    exponent_r <= shifted_difference_r > 15
                        ? 5'd15 : shifted_difference_r[4:0];
                    state <= ST_EXPONENT_QUANTIZE;
                end

                ST_EXPONENT_QUANTIZE: begin
                    unnormalized_r <= 16'd1 << (15 - exponent_r);
                    state <= ST_EXPONENT_ACCUMULATE;
                end

                ST_EXPONENT_ACCUMULATE: begin
                    exponent_mem[exponent_index[7:0]] <= unnormalized_r;
                    denominator_r <= denominator_r + {8'd0, unnormalized_r};
                    if (exponent_remaining_r == 9'd1) begin
                        state <= ST_RECIPROCAL_START;
                    end else begin
                        exponent_index <= exponent_index + 1'b1;
                        exponent_remaining_r <= exponent_remaining_r - 1'b1;
                        state <= ST_EXPONENT_READ;
                    end
                end

                ST_RECIPROCAL_START: begin
                    if (divider_input_ready) state <= ST_RECIPROCAL_WAIT;
                end

                ST_RECIPROCAL_WAIT: begin
                    if (divider_output_valid) begin
                        reciprocal_q31_r <= divider_quotient;
                        output_index_r <= 9'd0;
                        output_remaining_r <= item_count_r;
                        config_error <= config_error || divider_zero;
                        state <= ST_OUTPUT_READ;
                    end
                end

                ST_OUTPUT_READ: begin
                    output_exponent_ram_r <= exponent_mem[output_index_r[7:0]];
                    state <= ST_OUTPUT_LATCH;
                end

                ST_OUTPUT_LATCH: begin
                    output_exponent_r <= output_exponent_ram_r;
                    state <= ST_OUTPUT_OPERANDS;
                end

                ST_OUTPUT_OPERANDS: begin
                    multiply_exponent_r <= output_exponent_r;
                    multiply_reciprocal_r <= reciprocal_q31_r;
                    state <= ST_OUTPUT_MULTIPLY;
                end

                ST_OUTPUT_MULTIPLY: begin
                    probability_low_product_r <= probability_low_product;
                    probability_high_product_r <= probability_high_product;
                    state <= ST_OUTPUT_PRODUCT_ADD;
                end

                ST_OUTPUT_PRODUCT_ADD: begin
                    probability_product_r <= probability_low_product_r
                        + {probability_high_product_r[15:0], 16'd0};
                    state <= ST_OUTPUT_ROUND;
                end

                ST_OUTPUT_ROUND: begin
                    probability_q15 <= probability_product_r[31:16]
                        + 16'(probability_product_r[15]);
                    probability_last_r <= output_remaining_r == 9'd1;
                    state <= ST_OUTPUT_HOLD;
                end

                ST_OUTPUT_HOLD: begin
                    if (probability_ready) begin
                        if (probability_last) begin
                            state <= ST_IDLE;
                        end else begin
                            output_index_r <= output_index_r + 1'b1;
                            output_remaining_r <= output_remaining_r - 1'b1;
                            state <= ST_OUTPUT_READ;
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_ITEMS < 1 || MAX_ITEMS > 256) $fatal(1, "MAX_ITEMS must be in [1,256]");
        if (SCORE_BITS != 24) $fatal(1, "neuraedge_log_softmax currently requires SCORE_BITS=24");
    end
endmodule

`default_nettype wire
