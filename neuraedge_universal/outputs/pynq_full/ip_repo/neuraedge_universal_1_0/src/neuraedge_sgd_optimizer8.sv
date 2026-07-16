`timescale 1ns / 1ps
`default_nettype none

module neuraedge_sgd_optimizer8 #(
    parameter int LANES = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_counters,
    input  wire input_valid,
    output logic input_ready,
    input  wire [LANES*16-1:0] weight_vector,
    input  wire [LANES*24-1:0] gradient_vector,
    input  wire [LANES*24-1:0] velocity_vector,
    input  wire [17:0] learning_rate_q16,
    input  wire [16:0] momentum_q16,
    input  wire [16:0] weight_decay_q16,
    input  wire [4:0] update_right_shift,
    output logic output_valid,
    input  wire output_ready,
    output logic [LANES*16-1:0] updated_weight_vector,
    output logic [LANES*24-1:0] updated_velocity_vector,
    output logic config_error,
    output logic [31:0] update_count
);
    typedef enum logic [4:0] {
        ST_IDLE = 5'd0,
        ST_MOMENTUM_MULTIPLY = 5'd1,
        ST_MOMENTUM_ABSOLUTE = 5'd2,
        ST_MOMENTUM_ROUND = 5'd3,
        ST_MOMENTUM_SIGN = 5'd4,
        ST_DECAY_MULTIPLY = 5'd5,
        ST_DECAY_ABSOLUTE = 5'd6,
        ST_DECAY_ROUND = 5'd7,
        ST_DECAY_SIGN = 5'd8,
        ST_VELOCITY = 5'd9,
        ST_UPDATE_MULTIPLY = 5'd10,
        ST_UPDATE_ABSOLUTE = 5'd11,
        ST_UPDATE_ROUND = 5'd12,
        ST_UPDATE_SIGN = 5'd13,
        ST_UPDATE_SUBTRACT = 5'd14,
        ST_OUTPUT = 5'd15,
        ST_MOMENTUM_PRODUCT = 5'd16,
        ST_DECAY_PRODUCT = 5'd17,
        ST_UPDATE_PRODUCT = 5'd18,
        ST_VELOCITY_SUM = 5'd19,
        ST_VELOCITY_SATURATE = 5'd20,
        ST_MOMENTUM_SATURATE = 5'd21,
        ST_DECAY_SATURATE = 5'd22,
        ST_UPDATE_SATURATE = 5'd23,
        ST_MOMENTUM_BIAS = 5'd24,
        ST_DECAY_BIAS = 5'd25,
        ST_UPDATE_BIAS = 5'd26,
        ST_UPDATE_DIRECTION = 5'd27,
        ST_UPDATE_DIRECTION_SATURATE = 5'd28,
        ST_MOMENTUM_PRODUCT_RELAY = 5'd29,
        ST_DECAY_PRODUCT_RELAY = 5'd30,
        ST_UPDATE_PRODUCT_RELAY = 5'd31
    } state_t;

    state_t state;
    logic signed [15:0] weight_r [0:LANES-1];
    logic signed [23:0] gradient_r [0:LANES-1];
    logic signed [23:0] velocity_r [0:LANES-1];
    logic signed [23:0] next_velocity_r [0:LANES-1];
    logic signed [23:0] momentum_term_r [0:LANES-1];
    logic signed [23:0] decay_term_r [0:LANES-1];
    logic signed [24:0] two_term_sum_r [0:LANES-1];
    logic signed [25:0] velocity_sum_r [0:LANES-1];
    logic signed [41:0] product_r [0:LANES-1];
    logic product_negative_r [0:LANES-1];
    logic [41:0] product_magnitude_r [0:LANES-1];
    logic [41:0] rounded_magnitude_r [0:LANES-1];
    logic signed [41:0] signed_rounded_r [0:LANES-1];
    logic signed [41:0] update_value_r [0:LANES-1];
    logic signed [41:0] weight_difference_r [0:LANES-1];
    logic [17:0] learning_rate_r;
    logic [16:0] momentum_r;
    logic [16:0] weight_decay_r;
    logic [4:0] update_right_shift_r;
    logic [4:0] round_shift_counter_r;
    logic [41:0] update_rounding_bias_r;

    logic signed [23:0] multiplier_left [0:LANES-1];
    logic signed [17:0] multiplier_right [0:LANES-1];
    logic signed [23:0] multiplier_operand_left_r [0:LANES-1];
    logic signed [17:0] multiplier_operand_right_r [0:LANES-1];
    (* use_dsp = "yes" *) wire signed [41:0] multiplier_product [0:LANES-1];
    (* use_dsp = "yes" *) logic signed [41:0]
        multiplier_product_pipeline_r [0:LANES-1];

    function automatic logic signed [23:0] saturate_s24(input logic signed [41:0] value);
        begin
            if (value > 42'sd8388607) saturate_s24 = 24'sh7fffff;
            else if (value < -42'sd8388608) saturate_s24 = 24'sh800000;
            else saturate_s24 = value[23:0];
        end
    endfunction

    function automatic logic signed [15:0] saturate_s16(input logic signed [41:0] value);
        begin
            if (value > 42'sd32767) saturate_s16 = 16'sh7fff;
            else if (value < -42'sd32768) saturate_s16 = 16'sh8000;
            else saturate_s16 = value[15:0];
        end
    endfunction

    for (genvar lane = 0; lane < LANES; lane++) begin : g_optimizer_lane
        assign multiplier_product[lane] =
            multiplier_operand_left_r[lane] * multiplier_operand_right_r[lane];
    end

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
        for (int lane = 0; lane < LANES; lane++) begin
            multiplier_left[lane] = 24'sd0;
            multiplier_right[lane] = 18'sd0;
            unique case (state)
                ST_MOMENTUM_MULTIPLY: begin
                    multiplier_left[lane] = velocity_r[lane];
                    multiplier_right[lane] = $signed({1'b0, momentum_r});
                end
                ST_DECAY_MULTIPLY: begin
                    multiplier_left[lane] = {{8{weight_r[lane][15]}}, weight_r[lane]};
                    multiplier_right[lane] = $signed({1'b0, weight_decay_r});
                end
                ST_UPDATE_MULTIPLY: begin
                    multiplier_left[lane] = momentum_term_r[lane];
                    multiplier_right[lane] = $signed(learning_rate_r);
                end
                default: begin
                    multiplier_left[lane] = 24'sd0;
                    multiplier_right[lane] = 18'sd0;
                end
            endcase
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            updated_weight_vector <= '0;
            updated_velocity_vector <= '0;
            learning_rate_r <= 18'd0;
            momentum_r <= 17'd0;
            weight_decay_r <= 17'd0;
            update_right_shift_r <= 5'd16;
            round_shift_counter_r <= 5'd0;
            update_rounding_bias_r <= 42'd32768;
            config_error <= 1'b0;
            update_count <= 32'd0;
            for (int lane = 0; lane < LANES; lane++) begin
                weight_r[lane] <= 16'sd0;
                gradient_r[lane] <= 24'sd0;
                velocity_r[lane] <= 24'sd0;
                next_velocity_r[lane] <= 24'sd0;
                momentum_term_r[lane] <= 24'sd0;
                decay_term_r[lane] <= 24'sd0;
                two_term_sum_r[lane] <= 25'sd0;
                velocity_sum_r[lane] <= 26'sd0;
                product_r[lane] <= 42'sd0;
                product_negative_r[lane] <= 1'b0;
                product_magnitude_r[lane] <= 42'd0;
                rounded_magnitude_r[lane] <= 42'd0;
                signed_rounded_r[lane] <= 42'sd0;
                update_value_r[lane] <= 42'sd0;
                weight_difference_r[lane] <= 42'sd0;
                multiplier_operand_left_r[lane] <= 24'sd0;
                multiplier_operand_right_r[lane] <= 18'sd0;
                multiplier_product_pipeline_r[lane] <= 42'sd0;
            end
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        config_error <= learning_rate_q16 > 18'd131071
                            || momentum_q16 > 17'd65536
                            || weight_decay_q16 > 17'd65536;
                        learning_rate_r <= learning_rate_q16;
                        momentum_r <= momentum_q16;
                        weight_decay_r <= weight_decay_q16;
                        update_right_shift_r <= update_right_shift;
                        update_rounding_bias_r <= update_right_shift == 0
                            ? 42'd0 : (42'd1 << (update_right_shift - 1'b1));
                        for (int lane = 0; lane < LANES; lane++) begin
                            weight_r[lane] <= $signed(weight_vector[lane*16 +: 16]);
                            gradient_r[lane] <= $signed(gradient_vector[lane*24 +: 24]);
                            velocity_r[lane] <= $signed(velocity_vector[lane*24 +: 24]);
                        end
                        state <= ST_MOMENTUM_MULTIPLY;
                    end
                end

                ST_MOMENTUM_MULTIPLY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_operand_left_r[lane] <= multiplier_left[lane];
                        multiplier_operand_right_r[lane] <= multiplier_right[lane];
                    end
                    state <= ST_MOMENTUM_PRODUCT;
                end

                ST_MOMENTUM_PRODUCT: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_product_pipeline_r[lane]
                            <= multiplier_product[lane];
                    end
                    state <= ST_MOMENTUM_PRODUCT_RELAY;
                end

                ST_MOMENTUM_PRODUCT_RELAY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_r[lane] <= multiplier_product_pipeline_r[lane];
                    end
                    state <= ST_MOMENTUM_ABSOLUTE;
                end

                ST_MOMENTUM_ABSOLUTE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_negative_r[lane] <= product_r[lane][41];
                        product_magnitude_r[lane] <= product_r[lane][41]
                            ? -product_r[lane] : product_r[lane];
                    end
                    state <= ST_MOMENTUM_BIAS;
                end

                ST_MOMENTUM_BIAS: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        rounded_magnitude_r[lane]
                            <= product_magnitude_r[lane] + 42'd32768;
                    end
                    state <= ST_MOMENTUM_ROUND;
                end

                ST_MOMENTUM_ROUND: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        rounded_magnitude_r[lane] <= rounded_magnitude_r[lane] >> 16;
                    end
                    state <= ST_MOMENTUM_SIGN;
                end

                ST_MOMENTUM_SIGN: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        signed_rounded_r[lane] <= product_negative_r[lane]
                            ? -$signed(rounded_magnitude_r[lane])
                            : $signed(rounded_magnitude_r[lane]);
                    end
                    state <= ST_MOMENTUM_SATURATE;
                end

                ST_MOMENTUM_SATURATE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        momentum_term_r[lane] <= saturate_s24(signed_rounded_r[lane]);
                    end
                    state <= ST_DECAY_MULTIPLY;
                end

                ST_DECAY_MULTIPLY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_operand_left_r[lane] <= multiplier_left[lane];
                        multiplier_operand_right_r[lane] <= multiplier_right[lane];
                    end
                    state <= ST_DECAY_PRODUCT;
                end

                ST_DECAY_PRODUCT: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_product_pipeline_r[lane]
                            <= multiplier_product[lane];
                    end
                    state <= ST_DECAY_PRODUCT_RELAY;
                end

                ST_DECAY_PRODUCT_RELAY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_r[lane] <= multiplier_product_pipeline_r[lane];
                    end
                    state <= ST_DECAY_ABSOLUTE;
                end

                ST_DECAY_ABSOLUTE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_negative_r[lane] <= product_r[lane][41];
                        product_magnitude_r[lane] <= product_r[lane][41]
                            ? -product_r[lane] : product_r[lane];
                    end
                    state <= ST_DECAY_BIAS;
                end

                ST_DECAY_BIAS: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        rounded_magnitude_r[lane]
                            <= product_magnitude_r[lane] + 42'd32768;
                    end
                    state <= ST_DECAY_ROUND;
                end

                ST_DECAY_ROUND: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        rounded_magnitude_r[lane] <= rounded_magnitude_r[lane] >> 16;
                    end
                    state <= ST_DECAY_SIGN;
                end

                ST_DECAY_SIGN: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        signed_rounded_r[lane] <= product_negative_r[lane]
                            ? -$signed(rounded_magnitude_r[lane])
                            : $signed(rounded_magnitude_r[lane]);
                    end
                    state <= ST_DECAY_SATURATE;
                end

                ST_DECAY_SATURATE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        decay_term_r[lane] <= saturate_s24(signed_rounded_r[lane]);
                    end
                    state <= ST_VELOCITY;
                end

                ST_VELOCITY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        two_term_sum_r[lane] <=
                            {gradient_r[lane][23], gradient_r[lane]}
                            + {momentum_term_r[lane][23], momentum_term_r[lane]};
                    end
                    state <= ST_VELOCITY_SUM;
                end

                ST_VELOCITY_SUM: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        velocity_sum_r[lane] <=
                            {two_term_sum_r[lane][24], two_term_sum_r[lane]};
                    end
                    state <= ST_VELOCITY_SATURATE;
                end

                ST_VELOCITY_SATURATE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        next_velocity_r[lane] <= saturate_s24(
                            {{16{velocity_sum_r[lane][25]}}, velocity_sum_r[lane]});
                    end
                    state <= ST_UPDATE_DIRECTION;
                end

                ST_UPDATE_DIRECTION: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        two_term_sum_r[lane] <=
                            {next_velocity_r[lane][23], next_velocity_r[lane]}
                            + {decay_term_r[lane][23], decay_term_r[lane]};
                    end
                    state <= ST_UPDATE_DIRECTION_SATURATE;
                end

                ST_UPDATE_DIRECTION_SATURATE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        // The retained velocity excludes weight decay. This
                        // scratch register holds only the SGDW update direction.
                        momentum_term_r[lane] <= saturate_s24(
                            {{17{two_term_sum_r[lane][24]}}, two_term_sum_r[lane]});
                    end
                    state <= ST_UPDATE_MULTIPLY;
                end

                ST_UPDATE_MULTIPLY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_operand_left_r[lane] <= multiplier_left[lane];
                        multiplier_operand_right_r[lane] <= multiplier_right[lane];
                    end
                    state <= ST_UPDATE_PRODUCT;
                end

                ST_UPDATE_PRODUCT: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        multiplier_product_pipeline_r[lane]
                            <= multiplier_product[lane];
                    end
                    state <= ST_UPDATE_PRODUCT_RELAY;
                end

                ST_UPDATE_PRODUCT_RELAY: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_r[lane] <= multiplier_product_pipeline_r[lane];
                    end
                    state <= ST_UPDATE_ABSOLUTE;
                end

                ST_UPDATE_ABSOLUTE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        product_negative_r[lane] <= product_r[lane][41];
                        product_magnitude_r[lane] <= product_r[lane][41]
                            ? -product_r[lane] : product_r[lane];
                    end
                    state <= ST_UPDATE_BIAS;
                end

                ST_UPDATE_BIAS: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        rounded_magnitude_r[lane] <=
                            product_magnitude_r[lane] + update_rounding_bias_r;
                    end
                    round_shift_counter_r <= update_right_shift_r;
                    state <= ST_UPDATE_ROUND;
                end

                ST_UPDATE_ROUND: begin
                    if (round_shift_counter_r == 0) begin
                        state <= ST_UPDATE_SIGN;
                    end else begin
                        for (int lane = 0; lane < LANES; lane++) begin
                            rounded_magnitude_r[lane]
                                <= rounded_magnitude_r[lane] >> 1;
                        end
                        round_shift_counter_r <= round_shift_counter_r - 1'b1;
                        if (round_shift_counter_r == 1) state <= ST_UPDATE_SIGN;
                    end
                end

                ST_UPDATE_SIGN: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        update_value_r[lane] <= product_negative_r[lane]
                            ? -$signed(rounded_magnitude_r[lane])
                            : $signed(rounded_magnitude_r[lane]);
                    end
                    state <= ST_UPDATE_SUBTRACT;
                end

                ST_UPDATE_SUBTRACT: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        weight_difference_r[lane] <=
                            {{26{weight_r[lane][15]}}, weight_r[lane]}
                            - update_value_r[lane];
                        updated_velocity_vector[lane*24 +: 24] <= next_velocity_r[lane];
                    end
                    state <= ST_UPDATE_SATURATE;
                end

                ST_UPDATE_SATURATE: begin
                    for (int lane = 0; lane < LANES; lane++) begin
                        updated_weight_vector[lane*16 +: 16] <=
                            saturate_s16(weight_difference_r[lane]);
                    end
                    update_count <= update_count + LANES;
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: begin
                    if (output_ready) state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
            if (clear_counters) update_count <= 32'd0;
        end
    end

    initial begin
        if (LANES != 8) $fatal(1, "neuraedge_sgd_optimizer8 currently requires LANES=8");
    end
endmodule

`default_nettype wire
