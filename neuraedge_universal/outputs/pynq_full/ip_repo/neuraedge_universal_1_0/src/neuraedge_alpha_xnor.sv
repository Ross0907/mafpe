`timescale 1ns / 1ps
`default_nettype none

module neuraedge_alpha_xnor #(
    parameter int VECTOR_BITS = 256
) (
    input  wire clk,
    input  wire rst_n,
    input  wire input_valid,
    output logic input_ready,
    input  wire [VECTOR_BITS-1:0] query_spikes,
    input  wire [VECTOR_BITS-1:0] key_spikes,
    input  wire [8:0] active_bits,
    input  wire [8:0] alpha_q8,
    output logic output_valid,
    input  wire output_ready,
    output logic [17:0] score_q8,
    output logic [8:0] spike_matches,
    output logic [8:0] zero_matches,
    output logic config_error
);
    localparam int CHUNK_BITS = 32;
    localparam int CHUNKS = VECTOR_BITS / CHUNK_BITS;

    typedef enum logic [3:0] {
        ST_IDLE = 4'd0,
        ST_MASK_CHUNK = 4'd1,
        ST_ACCUMULATE = 4'd2,
        ST_SCORE_MULTIPLY = 4'd3,
        ST_SCORE_ADD = 4'd4,
        ST_OUTPUT = 4'd5,
        ST_SCORE_LATCH = 4'd6,
        ST_POPCOUNT = 4'd7,
        ST_POPCOUNT_PARTIAL = 4'd8,
        ST_VALIDATE = 4'd9,
        ST_POPCOUNT_HALF = 4'd10,
        ST_POPCOUNT_FINAL = 4'd11
    } state_t;

    state_t state;
    logic [VECTOR_BITS-1:0] query_r;
    logic [VECTOR_BITS-1:0] key_r;
    logic [8:0] active_bits_r;
    logic [8:0] alpha_q8_r;
    logic [$clog2(CHUNKS)-1:0] chunk_index;
    logic [31:0] spike_mask_r;
    logic [31:0] zero_mask_r;
    logic [8:0] spike_count_r;
    logic [8:0] zero_count_r;
    logic [5:0] spike_chunk_count_r;
    logic [5:0] zero_chunk_count_r;
    logic [2:0] spike_nibble_count_r [0:7];
    logic [2:0] zero_nibble_count_r [0:7];
    logic [3:0] spike_pair_count_r [0:3];
    logic [3:0] zero_pair_count_r [0:3];
    logic [4:0] spike_half_count_r [0:1];
    logic [4:0] zero_half_count_r [0:1];
    logic [8:0] final_spike_count_r;
    logic [8:0] final_zero_count_r;
    (* dont_touch = "yes" *) logic [8:0] score_spike_operand_r;
    (* dont_touch = "yes" *) logic [8:0] score_zero_operand_r;
    logic [17:0] alpha_product_r;

    (* use_dsp = "yes" *) wire [17:0] alpha_product =
        score_zero_operand_r * alpha_q8_r;

    function automatic logic [2:0] popcount4(input logic [3:0] value);
        logic [1:0] low_pair;
        logic [1:0] high_pair;
        begin
            low_pair = value[0] + value[1];
            high_pair = value[2] + value[3];
            popcount4 = low_pair + high_pair;
        end
    endfunction

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            query_r <= '0;
            key_r <= '0;
            active_bits_r <= 9'd0;
            alpha_q8_r <= 9'd0;
            chunk_index <= '0;
            spike_mask_r <= 32'd0;
            zero_mask_r <= 32'd0;
            spike_count_r <= 9'd0;
            zero_count_r <= 9'd0;
            spike_chunk_count_r <= 6'd0;
            zero_chunk_count_r <= 6'd0;
            for (int nibble = 0; nibble < 8; nibble++) begin
                spike_nibble_count_r[nibble] <= 3'd0;
                zero_nibble_count_r[nibble] <= 3'd0;
            end
            for (int pair = 0; pair < 4; pair++) begin
                spike_pair_count_r[pair] <= 4'd0;
                zero_pair_count_r[pair] <= 4'd0;
            end
            for (int half = 0; half < 2; half++) begin
                spike_half_count_r[half] <= 5'd0;
                zero_half_count_r[half] <= 5'd0;
            end
            final_spike_count_r <= 9'd0;
            final_zero_count_r <= 9'd0;
            score_spike_operand_r <= 9'd0;
            score_zero_operand_r <= 9'd0;
            alpha_product_r <= 18'd0;
            score_q8 <= 18'd0;
            spike_matches <= 9'd0;
            zero_matches <= 9'd0;
            config_error <= 1'b0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        query_r <= query_spikes;
                        key_r <= key_spikes;
                        active_bits_r <= active_bits;
                        alpha_q8_r <= alpha_q8;
                        state <= ST_VALIDATE;
                    end
                end

                ST_VALIDATE: begin
                        if (active_bits_r == 0
                                || active_bits_r > 9'(VECTOR_BITS)
                                || alpha_q8_r > 256) begin
                            config_error <= 1'b1;
                            score_q8 <= 18'd0;
                            spike_matches <= 9'd0;
                            zero_matches <= 9'd0;
                            state <= ST_OUTPUT;
                        end else begin
                            config_error <= 1'b0;
                            chunk_index <= '0;
                            spike_count_r <= 9'd0;
                            zero_count_r <= 9'd0;
                            state <= ST_MASK_CHUNK;
                        end
                end

                ST_MASK_CHUNK: begin
                    for (int bit_index = 0; bit_index < CHUNK_BITS; bit_index++) begin
                        logic [8:0] global_bit;
                        logic query_bit;
                        logic key_bit;
                        global_bit = {chunk_index, 5'd0} + 9'(bit_index);
                        query_bit = query_r[chunk_index*CHUNK_BITS + bit_index];
                        key_bit = key_r[chunk_index*CHUNK_BITS + bit_index];
                        spike_mask_r[bit_index] <= global_bit < active_bits_r
                            && query_bit && key_bit;
                        zero_mask_r[bit_index] <= global_bit < active_bits_r
                            && !query_bit && !key_bit;
                    end
                    state <= ST_POPCOUNT_PARTIAL;
                end

                ST_POPCOUNT_PARTIAL: begin
                    for (int nibble = 0; nibble < 8; nibble++) begin
                        spike_nibble_count_r[nibble] <=
                            popcount4(spike_mask_r[nibble*4 +: 4]);
                        zero_nibble_count_r[nibble] <=
                            popcount4(zero_mask_r[nibble*4 +: 4]);
                    end
                    state <= ST_POPCOUNT;
                end

                ST_POPCOUNT: begin
                    for (int pair = 0; pair < 4; pair++) begin
                        spike_pair_count_r[pair] <= spike_nibble_count_r[2*pair]
                            + spike_nibble_count_r[2*pair+1];
                        zero_pair_count_r[pair] <= zero_nibble_count_r[2*pair]
                            + zero_nibble_count_r[2*pair+1];
                    end
                    state <= ST_POPCOUNT_HALF;
                end

                ST_POPCOUNT_HALF: begin
                    spike_half_count_r[0] <= spike_pair_count_r[0]
                        + spike_pair_count_r[1];
                    spike_half_count_r[1] <= spike_pair_count_r[2]
                        + spike_pair_count_r[3];
                    zero_half_count_r[0] <= zero_pair_count_r[0]
                        + zero_pair_count_r[1];
                    zero_half_count_r[1] <= zero_pair_count_r[2]
                        + zero_pair_count_r[3];
                    state <= ST_POPCOUNT_FINAL;
                end

                ST_POPCOUNT_FINAL: begin
                    spike_chunk_count_r <= spike_half_count_r[0]
                        + spike_half_count_r[1];
                    zero_chunk_count_r <= zero_half_count_r[0]
                        + zero_half_count_r[1];
                    state <= ST_ACCUMULATE;
                end

                ST_ACCUMULATE: begin
                    logic [8:0] spike_total;
                    logic [8:0] zero_total;
                    spike_total = spike_count_r + {3'd0, spike_chunk_count_r};
                    zero_total = zero_count_r + {3'd0, zero_chunk_count_r};
                    spike_count_r <= spike_total;
                    zero_count_r <= zero_total;
                    if (chunk_index == $clog2(CHUNKS)'(CHUNKS - 1)) begin
                        final_spike_count_r <= spike_total;
                        final_zero_count_r <= zero_total;
                        spike_matches <= spike_total;
                        zero_matches <= zero_total;
                        state <= ST_SCORE_LATCH;
                    end else begin
                        chunk_index <= chunk_index + 1'b1;
                        state <= ST_MASK_CHUNK;
                    end
                end

                ST_SCORE_LATCH: begin
                    score_spike_operand_r <= final_spike_count_r;
                    score_zero_operand_r <= final_zero_count_r;
                    state <= ST_SCORE_MULTIPLY;
                end

                ST_SCORE_MULTIPLY: begin
                    alpha_product_r <= alpha_product;
                    state <= ST_SCORE_ADD;
                end

                ST_SCORE_ADD: begin
                    score_q8 <= {score_spike_operand_r, 8'd0} + alpha_product_r;
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: if (output_ready) state <= ST_IDLE;
                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (VECTOR_BITS != 256) $fatal(1, "neuraedge_alpha_xnor currently requires VECTOR_BITS=256");
        if (VECTOR_BITS % CHUNK_BITS != 0) $fatal(1, "VECTOR_BITS must be divisible by 32");
    end
endmodule

`default_nettype wire
