`timescale 1ns / 1ps
`default_nettype none

module neuraedge_isqrt64 (
    input  wire clk,
    input  wire rst_n,
    input  wire input_valid,
    output logic input_ready,
    input  wire [63:0] radicand,
    output logic output_valid,
    input  wire output_ready,
    output logic [31:0] root,
    output logic [32:0] remainder
);
    typedef enum logic [1:0] {
        ST_IDLE = 2'd0,
        ST_ROOT = 2'd1,
        ST_OUTPUT = 2'd2
    } state_t;

    state_t state;
    logic [63:0] radicand_r;
    logic [31:0] root_r;
    // Before the final digit, remainder < 2*partial_root+1 < 2^32. The final
    // remainder may need bit 32 and is written directly to the output port.
    logic [31:0] remainder_r;
    logic [5:0] pair_count;

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            radicand_r <= 64'd0;
            root_r <= 32'd0;
            remainder_r <= 32'd0;
            pair_count <= 6'd0;
            root <= 32'd0;
            remainder <= 33'd0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        radicand_r <= radicand;
                        root_r <= 32'd0;
                        remainder_r <= 32'd0;
                        pair_count <= 6'd0;
                        state <= ST_ROOT;
                    end
                end

                ST_ROOT: begin
                    logic [33:0] shifted_remainder;
                    logic [33:0] trial_divisor;
                    logic [33:0] trial_difference;
                    logic [32:0] next_remainder;
                    logic [31:0] next_root;
                    shifted_remainder = {remainder_r, radicand_r[63:62]};
                    trial_divisor = {root_r, 2'b01};
                    // One widened subtraction produces both the candidate
                    // remainder and its unsigned borrow. This removes the
                    // serial compare-then-subtract carry chains.
                    trial_difference = shifted_remainder - trial_divisor;
                    if (!trial_difference[33]) begin
                        next_remainder = trial_difference[32:0];
                        next_root = {root_r[30:0], 1'b1};
                    end else begin
                        next_remainder = shifted_remainder[32:0];
                        next_root = {root_r[30:0], 1'b0};
                    end
                    radicand_r <= {radicand_r[61:0], 2'b00};
                    remainder_r <= next_remainder[31:0];
                    root_r <= next_root;

                    if (pair_count == 6'd31) begin
                        root <= next_root;
                        remainder <= next_remainder;
                        state <= ST_OUTPUT;
                    end else begin
                        pair_count <= pair_count + 1'b1;
                    end
                end

                ST_OUTPUT: begin
                    if (output_ready) state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
