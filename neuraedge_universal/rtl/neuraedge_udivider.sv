`timescale 1ns / 1ps
`default_nettype none

module neuraedge_udivider #(
    parameter int NUMERATOR_BITS = 64,
    parameter int DENOMINATOR_BITS = 32
) (
    input  wire clk,
    input  wire rst_n,
    input  wire input_valid,
    output logic input_ready,
    input  wire [NUMERATOR_BITS-1:0] numerator,
    input  wire [DENOMINATOR_BITS-1:0] denominator,
    output logic output_valid,
    input  wire output_ready,
    output logic [NUMERATOR_BITS-1:0] quotient,
    output logic [DENOMINATOR_BITS-1:0] remainder,
    output logic divide_by_zero
);
    localparam int INDEX_BITS = $clog2(NUMERATOR_BITS);

    typedef enum logic [2:0] {
        ST_IDLE = 3'd0,
        ST_CAPTURE = 3'd1,
        ST_SHIFT = 3'd2,
        ST_SUBTRACT = 3'd3,
        ST_OUTPUT = 3'd4,
        ST_VALIDATE = 3'd5
    } state_t;

    state_t state;
    logic [NUMERATOR_BITS-1:0] numerator_r;
    logic [DENOMINATOR_BITS-1:0] denominator_r;
    logic [NUMERATOR_BITS-1:0] quotient_r;
    logic [DENOMINATOR_BITS-1:0] remainder_r;
    logic [DENOMINATOR_BITS:0] shifted_remainder_r;
    (* extract_enable = "no" *) logic [INDEX_BITS-1:0] bit_index;
    logic denominator_zero_r;

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            numerator_r <= '0;
            denominator_r <= '0;
            quotient_r <= '0;
            remainder_r <= '0;
            shifted_remainder_r <= '0;
            bit_index <= '0;
            denominator_zero_r <= 1'b0;
            quotient <= '0;
            remainder <= '0;
            divide_by_zero <= 1'b0;
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        numerator_r <= numerator;
                        denominator_r <= denominator;
                        state <= ST_CAPTURE;
                    end
                end

                ST_CAPTURE: begin
                    denominator_zero_r <= denominator_r == 0;
                    divide_by_zero <= denominator_r == 0;
                    quotient_r <= '0;
                    remainder_r <= '0;
                    state <= ST_VALIDATE;
                end

                ST_VALIDATE: begin
                    if (denominator_zero_r) begin
                        quotient <= {NUMERATOR_BITS{1'b1}};
                        remainder <= numerator_r[DENOMINATOR_BITS-1:0];
                        state <= ST_OUTPUT;
                    end else begin
                        bit_index <= $clog2(NUMERATOR_BITS)'(NUMERATOR_BITS - 1);
                        state <= ST_SHIFT;
                    end
                end

                ST_SHIFT: begin
                    shifted_remainder_r <= {
                        remainder_r,
                        numerator_r[bit_index]
                    };
                    state <= ST_SUBTRACT;
                end

                ST_SUBTRACT: begin
                    logic [DENOMINATOR_BITS-1:0] next_remainder;
                    logic [NUMERATOR_BITS-1:0] next_quotient;
                    next_quotient = quotient_r;
                    if (shifted_remainder_r >= {1'b0, denominator_r}) begin
                        next_remainder = shifted_remainder_r[DENOMINATOR_BITS-1:0]
                            - denominator_r;
                        next_quotient[bit_index] = 1'b1;
                    end else begin
                        next_remainder = shifted_remainder_r[DENOMINATOR_BITS-1:0];
                        next_quotient[bit_index] = 1'b0;
                    end
                    // A completed subtraction is strictly less than the
                    // denominator, so the stored remainder needs no guard bit.
                    remainder_r <= next_remainder;
                    quotient_r <= next_quotient;

                    if (bit_index == 0) begin
                        quotient <= next_quotient;
                        remainder <= next_remainder;
                        state <= ST_OUTPUT;
                    end else begin
                        bit_index <= bit_index - 1'b1;
                        state <= ST_SHIFT;
                    end
                end

                ST_OUTPUT: begin
                    if (output_ready) state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (NUMERATOR_BITS < DENOMINATOR_BITS) begin
            $fatal(1, "NUMERATOR_BITS must be at least DENOMINATOR_BITS");
        end
        if (NUMERATOR_BITS < 2 || (1 << INDEX_BITS) != NUMERATOR_BITS) begin
            $fatal(1, "NUMERATOR_BITS must be a power of two");
        end
    end
endmodule

`default_nettype wire
