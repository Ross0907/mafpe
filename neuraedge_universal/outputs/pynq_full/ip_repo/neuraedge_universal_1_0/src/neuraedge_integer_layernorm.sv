`timescale 1ns / 1ps
`default_nettype none

module neuraedge_integer_layernorm #(
    parameter int MAX_FEATURES = 2048,
    parameter int VALUE_BITS = 24
) (
    input  wire clk,
    input  wire rst_n,
    input  wire start_valid,
    output logic start_ready,
    input  wire [11:0] feature_count,
    input  wire [31:0] epsilon,
    input  wire sample_valid,
    output logic sample_ready,
    input  wire signed [VALUE_BITS-1:0] sample_value,
    input  wire signed [VALUE_BITS-1:0] sample_gamma_q15,
    input  wire signed [VALUE_BITS-1:0] sample_beta,
    input  wire sample_last,
    output logic result_valid,
    input  wire result_ready,
    output logic signed [31:0] result_value,
    output logic [11:0] result_index,
    output logic result_last,
    output logic config_error,
    output logic [31:0] cycle_count,
    output logic signed [31:0] debug_mean,
    output logic [31:0] debug_variance,
    output logic [31:0] debug_sigma,
    output logic [31:0] debug_reciprocal_q31
);
    // The normalization datapath is deliberately deeply registered.  Each state
    // performs one memory access, multiply, carry-chain operation, or comparison;
    // no state hides a multi-operation arithmetic expression from timing analysis.
    typedef enum logic [5:0] {
        ST_IDLE = 6'd0,
        ST_LOAD = 6'd1,
        ST_MEAN_START = 6'd2,
        ST_MEAN_WAIT = 6'd3,
        ST_VARIANCE_READ = 6'd4,
        ST_VARIANCE_CENTER = 6'd5,
        ST_VARIANCE_SQUARE = 6'd6,
        ST_VARIANCE_ACCUMULATE = 6'd7,
        ST_VARIANCE_START = 6'd8,
        ST_VARIANCE_WAIT = 6'd9,
        ST_SQRT_START = 6'd10,
        ST_SQRT_WAIT = 6'd11,
        ST_RECIPROCAL_START = 6'd12,
        ST_RECIPROCAL_WAIT = 6'd13,
        ST_OUTPUT_READ = 6'd14,
        ST_OUTPUT_CENTER = 6'd15,
        ST_OUTPUT_NORMALIZE_MULTIPLY = 6'd16,
        ST_OUTPUT_NORMALIZE_ABSOLUTE = 6'd17,
        ST_OUTPUT_NORMALIZE_ROUND = 6'd18,
        ST_OUTPUT_NORMALIZE_SIGN = 6'd19,
        ST_OUTPUT_GAMMA_MULTIPLY = 6'd20,
        ST_OUTPUT_SCALE_ABSOLUTE = 6'd21,
        ST_OUTPUT_SCALE_ROUND = 6'd22,
        ST_OUTPUT_SCALE_SIGN = 6'd23,
        ST_OUTPUT_BETA = 6'd24,
        ST_OUTPUT_SATURATE = 6'd25,
        ST_OUTPUT_HOLD = 6'd26,
        ST_VARIANCE_LATCH = 6'd27,
        ST_OUTPUT_LATCH = 6'd28,
        ST_OUTPUT_NORMALIZE_LATCH = 6'd29,
        ST_OUTPUT_GAMMA_LATCH = 6'd30,
        ST_VARIANCE_SQUARE_LATCH = 6'd31,
        ST_OUTPUT_NORMALIZE_OPERANDS = 6'd32,
        ST_OUTPUT_NORMALIZE_CLAMP = 6'd33,
        ST_OUTPUT_GAMMA_OPERANDS = 6'd34,
        ST_OUTPUT_GAMMA_CROSS = 6'd35,
        ST_OUTPUT_GAMMA_LOW = 6'd36,
        ST_MEAN_ADJUST = 6'd37,
        ST_MEAN_COMMIT = 6'd38,
        ST_VARIANCE_SQUARE_PARTIAL = 6'd39,
        ST_VARIANCE_SQUARE_ADD = 6'd40,
        ST_MEAN_PREPARE = 6'd41,
        ST_VARIANCE_ACCUMULATE_HIGH = 6'd42,
        ST_OUTPUT_NORMALIZE_ABSOLUTE_HIGH = 6'd43,
        ST_MEAN_ABSOLUTE_HIGH = 6'd44
    } state_t;

    state_t state;
    (* ram_style = "block" *) logic signed [VALUE_BITS-1:0] value_mem [0:MAX_FEATURES-1];
    (* ram_style = "block" *) logic signed [VALUE_BITS-1:0] gamma_mem [0:MAX_FEATURES-1];
    (* ram_style = "block" *) logic signed [VALUE_BITS-1:0] beta_mem [0:MAX_FEATURES-1];

    logic [11:0] feature_count_r;
    logic [31:0] epsilon_r;
    logic [11:0] load_index;
    logic [11:0] samples_remaining_r;
    logic [11:0] variance_index;
    logic [11:0] variance_remaining_r;
    logic [11:0] output_index_r;
    logic [11:0] output_remaining_r;
    logic signed [47:0] sum_r;
    logic [63:0] mean_numerator_r;
    logic [31:0] mean_absolute_low_r;
    logic mean_absolute_carry_r;
    logic [63:0] variance_sum_r;
    logic [31:0] variance_sum_low_r;
    logic variance_sum_carry_r;
    logic signed [31:0] mean_r;
    logic mean_negative_r;
    logic mean_has_remainder_r;
    logic [31:0] mean_quotient_r;
    logic [31:0] mean_magnitude_r;
    logic [31:0] variance_r;
    logic [31:0] sigma_r;
    logic [31:0] reciprocal_q31_r;

    logic signed [VALUE_BITS-1:0] output_value_r;
    logic signed [VALUE_BITS-1:0] output_gamma_r;
    logic signed [VALUE_BITS-1:0] output_beta_r;
    logic signed [VALUE_BITS-1:0] output_value_ram_r;
    logic signed [VALUE_BITS-1:0] output_gamma_ram_r;
    logic signed [VALUE_BITS-1:0] output_beta_ram_r;
    logic signed [VALUE_BITS-1:0] variance_value_ram_r;
    logic signed [VALUE_BITS-1:0] variance_value_r;
    logic signed [24:0] variance_centered_r;
    logic [23:0] variance_magnitude_r;
    logic [35:0] variance_square_low_product_r;
    logic [35:0] variance_square_high_product_r;
    logic [49:0] variance_square_r;
    logic [49:0] variance_square_latched_r;
    (* dont_touch = "yes" *) logic signed [24:0] output_centered_r;
    logic signed [24:0] normalized_centered_operand_r;
    logic [31:0] normalized_reciprocal_operand_r;
    logic signed [41:0] normalized_low_product_r;
    logic signed [41:0] normalized_high_product_r;
    logic signed [57:0] normalized_product_r;
    logic normalized_negative_r;
    logic [31:0] normalized_magnitude_low_r;
    logic normalized_magnitude_carry_r;
    logic [63:0] normalized_magnitude_r;
    logic [63:0] normalized_rounded_magnitude_r;
    logic [31:0] normalized_clamped_magnitude_r;
    (* dont_touch = "yes" *) logic signed [31:0] normalized_r;
    logic signed [31:0] gamma_normalized_operand_r;
    logic signed [VALUE_BITS-1:0] gamma_scale_operand_r;
    logic signed [33:0] gamma_partial_ll_r;
    logic signed [24:0] gamma_partial_lh_r;
    logic signed [32:0] gamma_partial_hl_r;
    logic signed [23:0] gamma_partial_hh_r;
    logic signed [33:0] gamma_cross_r;
    logic signed [49:0] gamma_low_cross_r;
    logic signed [55:0] gamma_product_r;
    logic scale_negative_r;
    logic [55:0] scale_magnitude_r;
    logic [55:0] scale_rounded_magnitude_r;
    logic signed [55:0] scaled_r;
    logic signed [63:0] scaled_beta_r;
    logic result_last_r;

    // A difference between two signed 24-bit values fits in 25 bits and its
    // magnitude fits in 24 unsigned bits. Two 24 x 12 DSP products therefore
    // form the exact square without a multi-DSP combinational cascade.
    (* use_dsp = "yes" *) wire [35:0] variance_square_low_product =
        variance_magnitude_r[23:0] * variance_magnitude_r[11:0];
    (* use_dsp = "yes" *) wire [35:0] variance_square_high_product =
        variance_magnitude_r[23:0] * variance_magnitude_r[23:12];

    // A 24-bit sample minus its mean needs only 25 signed bits. Splitting the
    // unsigned Q31 reciprocal into two 16-bit limbs lets each exact partial
    // product fit one DSP48E1 without an unregistered PCIN cascade.
    (* use_dsp = "yes" *) wire signed [41:0] normalized_low_product =
        normalized_centered_operand_r
        * $signed({1'b0, normalized_reciprocal_operand_r[15:0]});
    (* use_dsp = "yes" *) wire signed [41:0] normalized_high_product =
        normalized_centered_operand_r
        * $signed({1'b0, normalized_reciprocal_operand_r[31:16]});

    // The signed 32 x 24 gamma multiply is four exact base-2^16 limb
    // products. Low limbs are made explicitly positive by a leading zero;
    // high limbs retain the sign of the original two's-complement operand.
    wire signed [16:0] gamma_normalized_low =
        $signed({1'b0, gamma_normalized_operand_r[15:0]});
    wire signed [15:0] gamma_normalized_high =
        $signed(gamma_normalized_operand_r[31:16]);
    wire signed [16:0] gamma_scale_low =
        $signed({1'b0, gamma_scale_operand_r[15:0]});
    wire signed [7:0] gamma_scale_high =
        $signed(gamma_scale_operand_r[23:16]);
    (* use_dsp = "yes" *) wire signed [33:0] gamma_partial_ll =
        gamma_normalized_low * gamma_scale_low;
    (* use_dsp = "yes" *) wire signed [24:0] gamma_partial_lh =
        gamma_normalized_low * gamma_scale_high;
    (* use_dsp = "yes" *) wire signed [32:0] gamma_partial_hl =
        gamma_normalized_high * gamma_scale_low;
    (* use_dsp = "yes" *) wire signed [23:0] gamma_partial_hh =
        gamma_normalized_high * gamma_scale_high;

    logic divider_input_valid;
    wire divider_input_ready;
    logic [63:0] divider_numerator;
    logic [31:0] divider_denominator;
    wire divider_output_valid;
    logic divider_output_ready;
    wire [63:0] divider_quotient;
    wire [31:0] divider_remainder;
    wire divider_zero;

    logic sqrt_input_valid;
    wire sqrt_input_ready;
    logic [63:0] sqrt_radicand;
    wire sqrt_output_valid;
    logic sqrt_output_ready;
    wire [31:0] sqrt_root;

    neuraedge_udivider u_divider (
        .clk(clk), .rst_n(rst_n),
        .input_valid(divider_input_valid), .input_ready(divider_input_ready),
        .numerator(divider_numerator), .denominator(divider_denominator),
        .output_valid(divider_output_valid), .output_ready(divider_output_ready),
        .quotient(divider_quotient), .remainder(divider_remainder),
        .divide_by_zero(divider_zero)
    );

    neuraedge_isqrt64 u_sqrt (
        .clk(clk), .rst_n(rst_n),
        .input_valid(sqrt_input_valid), .input_ready(sqrt_input_ready),
        .radicand(sqrt_radicand), .output_valid(sqrt_output_valid),
        .output_ready(sqrt_output_ready), .root(sqrt_root),
        .remainder()
    );

    function automatic logic signed [31:0] saturate_s32(
        input logic signed [63:0] value
    );
        begin
            if (value[63:32] == {32{value[31]}}) saturate_s32 = value[31:0];
            else if (value[63]) saturate_s32 = 32'sh80000000;
            else saturate_s32 = 32'sh7fffffff;
        end
    endfunction

    always_comb begin
        start_ready = state == ST_IDLE;
        sample_ready = state == ST_LOAD;
        result_valid = state == ST_OUTPUT_HOLD;
        result_index = output_index_r;
        result_last = result_last_r;

        divider_input_valid = state == ST_MEAN_START
            || state == ST_VARIANCE_START || state == ST_RECIPROCAL_START;
        divider_output_ready = state == ST_MEAN_WAIT
            || state == ST_VARIANCE_WAIT || state == ST_RECIPROCAL_WAIT;
        divider_numerator = 64'd0;
        divider_denominator = {20'd0, feature_count_r};
        if (state == ST_MEAN_START) begin
            divider_numerator = mean_numerator_r;
        end else if (state == ST_VARIANCE_START) begin
            divider_numerator = variance_sum_r;
        end else if (state == ST_RECIPROCAL_START) begin
            divider_numerator = 64'd2147483647;
            divider_denominator = sigma_r;
        end

        sqrt_input_valid = state == ST_SQRT_START;
        sqrt_output_ready = state == ST_SQRT_WAIT;
        sqrt_radicand = {32'd0, variance_r} + {32'd0, epsilon_r};

        debug_mean = mean_r;
        debug_variance = variance_r;
        debug_sigma = sigma_r;
        debug_reciprocal_q31 = reciprocal_q31_r;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            feature_count_r <= 12'd0;
            epsilon_r <= 32'd0;
            load_index <= 12'd0;
            samples_remaining_r <= 12'd0;
            variance_index <= 12'd0;
            variance_remaining_r <= 12'd0;
            output_index_r <= 12'd0;
            output_remaining_r <= 12'd0;
            sum_r <= 48'sd0;
            mean_numerator_r <= 64'd0;
            mean_absolute_low_r <= 32'd0;
            mean_absolute_carry_r <= 1'b0;
            variance_sum_r <= 64'd0;
            variance_sum_low_r <= 32'd0;
            variance_sum_carry_r <= 1'b0;
            mean_r <= 32'sd0;
            mean_negative_r <= 1'b0;
            mean_has_remainder_r <= 1'b0;
            mean_quotient_r <= 32'd0;
            mean_magnitude_r <= 32'd0;
            variance_r <= 32'd0;
            sigma_r <= 32'd1;
            reciprocal_q31_r <= 32'd0;
            output_value_r <= '0;
            output_gamma_r <= '0;
            output_beta_r <= '0;
            output_value_ram_r <= '0;
            output_gamma_ram_r <= '0;
            output_beta_ram_r <= '0;
            variance_value_ram_r <= '0;
            variance_value_r <= '0;
            variance_centered_r <= 25'sd0;
            variance_magnitude_r <= 24'd0;
            variance_square_low_product_r <= 36'd0;
            variance_square_high_product_r <= 36'd0;
            variance_square_r <= 50'd0;
            variance_square_latched_r <= 50'd0;
            output_centered_r <= 25'sd0;
            normalized_centered_operand_r <= 25'sd0;
            normalized_reciprocal_operand_r <= 32'd0;
            normalized_low_product_r <= 42'sd0;
            normalized_high_product_r <= 42'sd0;
            normalized_product_r <= 58'sd0;
            normalized_negative_r <= 1'b0;
            normalized_magnitude_low_r <= 32'd0;
            normalized_magnitude_carry_r <= 1'b0;
            normalized_magnitude_r <= 64'd0;
            normalized_rounded_magnitude_r <= 64'd0;
            normalized_clamped_magnitude_r <= 32'd0;
            normalized_r <= 32'sd0;
            gamma_normalized_operand_r <= 32'sd0;
            gamma_scale_operand_r <= '0;
            gamma_partial_ll_r <= 34'sd0;
            gamma_partial_lh_r <= 25'sd0;
            gamma_partial_hl_r <= 33'sd0;
            gamma_partial_hh_r <= 24'sd0;
            gamma_cross_r <= 34'sd0;
            gamma_low_cross_r <= 50'sd0;
            gamma_product_r <= 56'sd0;
            scale_negative_r <= 1'b0;
            scale_magnitude_r <= 56'd0;
            scale_rounded_magnitude_r <= 56'd0;
            scaled_r <= 56'sd0;
            scaled_beta_r <= 64'sd0;
            result_last_r <= 1'b0;
            result_value <= 32'sd0;
            config_error <= 1'b0;
            cycle_count <= 32'd0;
        end else begin
            if (state != ST_IDLE) cycle_count <= cycle_count + 1'b1;
            unique case (state)
                ST_IDLE: begin
                    if (start_valid) begin
                        cycle_count <= 32'd0;
                        config_error <= feature_count == 0
                            || feature_count > 12'(MAX_FEATURES);
                        feature_count_r <= feature_count;
                        epsilon_r <= epsilon;
                        // Clearing on every accepted start keeps feature-count
                        // validation out of the 48-bit sum register enable.
                        sum_r <= 48'sd0;
                        if (feature_count == 0
                                || feature_count > 12'(MAX_FEATURES)) begin
                            state <= ST_IDLE;
                        end else begin
                            load_index <= 12'd0;
                            samples_remaining_r <= feature_count;
                            state <= ST_LOAD;
                        end
                    end
                end

                ST_LOAD: begin
                    if (sample_valid) begin
                        logic signed [47:0] sample_extended;
                        logic expected_last;
                        sample_extended = {{(48-VALUE_BITS){sample_value[VALUE_BITS-1]}}, sample_value};
                        expected_last = samples_remaining_r == 12'd1;
                        value_mem[load_index[10:0]] <= sample_value;
                        gamma_mem[load_index[10:0]] <= sample_gamma_q15;
                        beta_mem[load_index[10:0]] <= sample_beta;
                        sum_r <= sum_r + sample_extended;
                        if (sample_last != expected_last) begin
                            config_error <= 1'b1;
                        end
                        if (expected_last) begin
                            state <= ST_MEAN_PREPARE;
                        end else if (sample_last) begin
                            config_error <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            load_index <= load_index + 1'b1;
                            samples_remaining_r <= samples_remaining_r - 1'b1;
                        end
                    end
                end

                ST_MEAN_PREPARE: begin
                    logic [32:0] absolute_low;
                    if (sum_r < 0) begin
                        absolute_low = {1'b0, ~sum_r[31:0]} + 1'b1;
                        mean_absolute_low_r <= absolute_low[31:0];
                        mean_absolute_carry_r <= absolute_low[32];
                    end else begin
                        mean_absolute_low_r <= sum_r[31:0];
                        mean_absolute_carry_r <= 1'b0;
                    end
                    state <= ST_MEAN_ABSOLUTE_HIGH;
                end

                ST_MEAN_ABSOLUTE_HIGH: begin
                    logic [15:0] absolute_high;
                    if (sum_r < 0) begin
                        absolute_high = ~sum_r[47:32]
                            + 16'(mean_absolute_carry_r);
                        mean_numerator_r <= {
                            16'd0, absolute_high, mean_absolute_low_r};
                    end else begin
                        mean_numerator_r <= {
                            16'd0, sum_r[47:32], mean_absolute_low_r};
                    end
                    state <= ST_MEAN_START;
                end

                ST_MEAN_START: begin
                    if (divider_input_ready) state <= ST_MEAN_WAIT;
                end

                ST_MEAN_WAIT: begin
                    if (divider_output_valid) begin
                        mean_negative_r <= sum_r < 0;
                        mean_has_remainder_r <= divider_remainder != 0;
                        mean_quotient_r <= divider_quotient[31:0];
                        config_error <= config_error || divider_zero;
                        state <= ST_MEAN_ADJUST;
                    end
                end

                ST_MEAN_ADJUST: begin
                    mean_magnitude_r <= mean_quotient_r
                        + (mean_negative_r && mean_has_remainder_r);
                    state <= ST_MEAN_COMMIT;
                end

                ST_MEAN_COMMIT: begin
                        mean_r <= mean_negative_r
                            ? -$signed(mean_magnitude_r)
                            : $signed(mean_magnitude_r);
                        variance_index <= 12'd0;
                        variance_remaining_r <= feature_count_r;
                        variance_sum_r <= 64'd0;
                        state <= ST_VARIANCE_READ;
                end

                ST_VARIANCE_READ: begin
                    variance_value_ram_r <= value_mem[variance_index[10:0]];
                    state <= ST_VARIANCE_LATCH;
                end

                ST_VARIANCE_LATCH: begin
                    variance_value_r <= variance_value_ram_r;
                    state <= ST_VARIANCE_CENTER;
                end

                ST_VARIANCE_CENTER: begin
                    variance_centered_r
                        <= $signed({variance_value_r[23], variance_value_r})
                            - $signed({mean_r[31], mean_r[23:0]});
                    state <= ST_VARIANCE_SQUARE;
                end

                ST_VARIANCE_SQUARE: begin
                    variance_magnitude_r <= variance_centered_r[24]
                        ? (~variance_centered_r[23:0] + 1'b1)
                        : variance_centered_r[23:0];
                    state <= ST_VARIANCE_SQUARE_PARTIAL;
                end

                ST_VARIANCE_SQUARE_PARTIAL: begin
                    variance_square_low_product_r <= variance_square_low_product;
                    variance_square_high_product_r <= variance_square_high_product;
                    state <= ST_VARIANCE_SQUARE_ADD;
                end

                ST_VARIANCE_SQUARE_ADD: begin
                    variance_square_r <= {14'd0, variance_square_low_product_r}
                        + {2'd0, variance_square_high_product_r, 12'd0};
                    state <= ST_VARIANCE_SQUARE_LATCH;
                end

                ST_VARIANCE_SQUARE_LATCH: begin
                    variance_square_latched_r <= variance_square_r;
                    state <= ST_VARIANCE_ACCUMULATE;
                end

                ST_VARIANCE_ACCUMULATE: begin
                    logic [32:0] low_sum;
                    low_sum = {1'b0, variance_sum_r[31:0]}
                        + {1'b0, variance_square_latched_r[31:0]};
                    variance_sum_low_r <= low_sum[31:0];
                    variance_sum_carry_r <= low_sum[32];
                    state <= ST_VARIANCE_ACCUMULATE_HIGH;
                end

                ST_VARIANCE_ACCUMULATE_HIGH: begin
                    logic [32:0] high_sum;
                    high_sum = {1'b0, variance_sum_r[63:32]}
                        + {1'b0, 14'd0, variance_square_latched_r[49:32]}
                        + variance_sum_carry_r;
                    variance_sum_r <= {high_sum[31:0], variance_sum_low_r};
                    if (high_sum[32]) config_error <= 1'b1;
                    if (variance_remaining_r == 12'd1) begin
                        state <= ST_VARIANCE_START;
                    end else begin
                        variance_index <= variance_index + 1'b1;
                        variance_remaining_r <= variance_remaining_r - 1'b1;
                        state <= ST_VARIANCE_READ;
                    end
                end

                ST_VARIANCE_START: begin
                    if (divider_input_ready) state <= ST_VARIANCE_WAIT;
                end

                ST_VARIANCE_WAIT: begin
                    if (divider_output_valid) begin
                        variance_r <= divider_quotient[31:0];
                        config_error <= config_error || divider_zero;
                        state <= ST_SQRT_START;
                    end
                end

                ST_SQRT_START: begin
                    if (sqrt_input_ready) state <= ST_SQRT_WAIT;
                end

                ST_SQRT_WAIT: begin
                    if (sqrt_output_valid) begin
                        sigma_r <= sqrt_root == 0 ? 32'd1 : sqrt_root;
                        state <= ST_RECIPROCAL_START;
                    end
                end

                ST_RECIPROCAL_START: begin
                    if (divider_input_ready) state <= ST_RECIPROCAL_WAIT;
                end

                ST_RECIPROCAL_WAIT: begin
                    if (divider_output_valid) begin
                        reciprocal_q31_r <= divider_quotient[31:0];
                        output_index_r <= 12'd0;
                        output_remaining_r <= feature_count_r;
                        config_error <= config_error || divider_zero;
                        state <= ST_OUTPUT_READ;
                    end
                end

                ST_OUTPUT_READ: begin
                    output_value_ram_r <= value_mem[output_index_r[10:0]];
                    output_gamma_ram_r <= gamma_mem[output_index_r[10:0]];
                    output_beta_ram_r <= beta_mem[output_index_r[10:0]];
                    state <= ST_OUTPUT_LATCH;
                end

                ST_OUTPUT_LATCH: begin
                    output_value_r <= output_value_ram_r;
                    output_gamma_r <= output_gamma_ram_r;
                    output_beta_r <= output_beta_ram_r;
                    state <= ST_OUTPUT_CENTER;
                end

                ST_OUTPUT_CENTER: begin
                    output_centered_r
                        <= $signed({output_value_r[23], output_value_r})
                            - $signed({mean_r[31], mean_r[23:0]});
                    state <= ST_OUTPUT_NORMALIZE_OPERANDS;
                end

                ST_OUTPUT_NORMALIZE_OPERANDS: begin
                    normalized_centered_operand_r <= output_centered_r;
                    normalized_reciprocal_operand_r <= reciprocal_q31_r;
                    state <= ST_OUTPUT_NORMALIZE_MULTIPLY;
                end

                ST_OUTPUT_NORMALIZE_MULTIPLY: begin
                    normalized_low_product_r <= normalized_low_product;
                    normalized_high_product_r <= normalized_high_product;
                    state <= ST_OUTPUT_NORMALIZE_LATCH;
                end

                ST_OUTPUT_NORMALIZE_LATCH: begin
                    normalized_product_r <=
                        {{16{normalized_low_product_r[41]}}, normalized_low_product_r}
                        + ($signed({{16{normalized_high_product_r[41]}},
                            normalized_high_product_r}) <<< 16);
                    state <= ST_OUTPUT_NORMALIZE_ABSOLUTE;
                end

                ST_OUTPUT_NORMALIZE_ABSOLUTE: begin
                    logic [32:0] magnitude_low;
                    normalized_negative_r <= normalized_product_r[57];
                    magnitude_low = normalized_product_r[57]
                        ? {1'b0, ~normalized_product_r[31:0]} + 1'b1
                        : {1'b0, normalized_product_r[31:0]};
                    normalized_magnitude_low_r <= magnitude_low[31:0];
                    normalized_magnitude_carry_r <= magnitude_low[32];
                    state <= ST_OUTPUT_NORMALIZE_ABSOLUTE_HIGH;
                end

                ST_OUTPUT_NORMALIZE_ABSOLUTE_HIGH: begin
                    logic [25:0] magnitude_high;
                    magnitude_high = normalized_negative_r
                        ? ~normalized_product_r[57:32]
                            + 26'(normalized_magnitude_carry_r)
                        : normalized_product_r[57:32];
                    normalized_magnitude_r <= {
                        6'd0, magnitude_high, normalized_magnitude_low_r};
                    state <= ST_OUTPUT_NORMALIZE_ROUND;
                end

                ST_OUTPUT_NORMALIZE_ROUND: begin
                    normalized_rounded_magnitude_r <=
                        (normalized_magnitude_r + 64'd1073741824) >> 31;
                    state <= ST_OUTPUT_NORMALIZE_CLAMP;
                end

                ST_OUTPUT_NORMALIZE_CLAMP: begin
                    if (normalized_negative_r) begin
                        normalized_clamped_magnitude_r <=
                            normalized_rounded_magnitude_r > 64'h0000000080000000
                            ? 32'h80000000
                            : normalized_rounded_magnitude_r[31:0];
                    end else begin
                        normalized_clamped_magnitude_r <=
                            normalized_rounded_magnitude_r > 64'h000000007fffffff
                            ? 32'h7fffffff
                            : normalized_rounded_magnitude_r[31:0];
                    end
                    state <= ST_OUTPUT_NORMALIZE_SIGN;
                end

                ST_OUTPUT_NORMALIZE_SIGN: begin
                    if (normalized_negative_r) begin
                        normalized_r <= normalized_clamped_magnitude_r[31]
                            ? 32'sh80000000
                            : -$signed(normalized_clamped_magnitude_r);
                    end else begin
                        normalized_r <= $signed(normalized_clamped_magnitude_r);
                    end
                    state <= ST_OUTPUT_GAMMA_OPERANDS;
                end

                ST_OUTPUT_GAMMA_OPERANDS: begin
                    gamma_normalized_operand_r <= normalized_r;
                    gamma_scale_operand_r <= output_gamma_r;
                    state <= ST_OUTPUT_GAMMA_MULTIPLY;
                end

                ST_OUTPUT_GAMMA_MULTIPLY: begin
                    gamma_partial_ll_r <= gamma_partial_ll;
                    gamma_partial_lh_r <= gamma_partial_lh;
                    gamma_partial_hl_r <= gamma_partial_hl;
                    gamma_partial_hh_r <= gamma_partial_hh;
                    state <= ST_OUTPUT_GAMMA_CROSS;
                end

                ST_OUTPUT_GAMMA_CROSS: begin
                    gamma_cross_r <=
                        {{9{gamma_partial_lh_r[24]}}, gamma_partial_lh_r}
                        + {{1{gamma_partial_hl_r[32]}}, gamma_partial_hl_r};
                    state <= ST_OUTPUT_GAMMA_LOW;
                end

                ST_OUTPUT_GAMMA_LOW: begin
                    gamma_low_cross_r <=
                        {{16{gamma_partial_ll_r[33]}}, gamma_partial_ll_r}
                        + ($signed({{16{gamma_cross_r[33]}}, gamma_cross_r}) <<< 16);
                    state <= ST_OUTPUT_GAMMA_LATCH;
                end

                ST_OUTPUT_GAMMA_LATCH: begin
                    gamma_product_r <=
                        {{6{gamma_low_cross_r[49]}}, gamma_low_cross_r}
                        + ($signed({{32{gamma_partial_hh_r[23]}},
                            gamma_partial_hh_r}) <<< 32);
                    state <= ST_OUTPUT_SCALE_ABSOLUTE;
                end

                ST_OUTPUT_SCALE_ABSOLUTE: begin
                    scale_negative_r <= gamma_product_r[55];
                    scale_magnitude_r <= gamma_product_r[55]
                        ? -gamma_product_r : gamma_product_r;
                    state <= ST_OUTPUT_SCALE_ROUND;
                end

                ST_OUTPUT_SCALE_ROUND: begin
                    scale_rounded_magnitude_r <=
                        (scale_magnitude_r + 56'd16384) >> 15;
                    state <= ST_OUTPUT_SCALE_SIGN;
                end

                ST_OUTPUT_SCALE_SIGN: begin
                    scaled_r <= scale_negative_r
                        ? -$signed(scale_rounded_magnitude_r)
                        : $signed(scale_rounded_magnitude_r);
                    state <= ST_OUTPUT_BETA;
                end

                ST_OUTPUT_BETA: begin
                    scaled_beta_r <= {{8{scaled_r[55]}}, scaled_r}
                        + {{(64-VALUE_BITS){output_beta_r[VALUE_BITS-1]}}, output_beta_r};
                    state <= ST_OUTPUT_SATURATE;
                end

                ST_OUTPUT_SATURATE: begin
                    result_value <= saturate_s32(scaled_beta_r);
                    result_last_r <= output_remaining_r == 12'd1;
                    state <= ST_OUTPUT_HOLD;
                end

                ST_OUTPUT_HOLD: begin
                    if (result_ready) begin
                        if (result_last) begin
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
        if (MAX_FEATURES < 1 || MAX_FEATURES > 2048) $fatal(1, "MAX_FEATURES must be in [1,2048]");
        if (VALUE_BITS != 24) $fatal(1, "neuraedge_integer_layernorm currently requires VALUE_BITS=24");
    end
endmodule

`default_nettype wire
