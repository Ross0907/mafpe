`timescale 1ns / 1ps
`default_nettype none

// Computes two exact unsigned 8-bit products in one 25 x 18 multiplier.
// The upper weight starts at bit 17, leaving bit 16 as a guard between the
// two 16-bit products. Since 255 * 255 is smaller than 2^16, the low product
// can never carry into the high product field.
module neuraedge_mac2_u8 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,
    input  wire [7:0]   activation,
    input  wire [7:0]   weight_lo,
    input  wire [7:0]   weight_hi,
    output wire         product_valid,
    output logic [15:0] product_lo,
    output logic [15:0] product_hi
);
    logic signed [24:0] packed_weight_r;
    logic signed [17:0] extended_activation_r;
    logic signed [47:0] unsigned_correction_r;
    (* dont_touch = "yes" *) logic input_valid_r;
    (* dont_touch = "yes" *) logic dsp_product_valid_r;
    (* dont_touch = "yes" *) logic product_valid_r;
    /* verilator lint_off UNUSEDSIGNAL */
    (* use_dsp = "yes" *) logic signed [47:0] packed_product;
    /* verilator lint_on UNUSEDSIGNAL */

    assign product_valid = product_valid_r;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            packed_weight_r <= '0;
            extended_activation_r <= '0;
            unsigned_correction_r <= '0;
            input_valid_r <= 1'b0;
            dsp_product_valid_r <= 1'b0;
            packed_product <= '0;
            product_valid_r <= 1'b0;
            product_lo <= 16'd0;
            product_hi <= 16'd0;
        end else begin
            input_valid_r <= enable;
            dsp_product_valid_r <= input_valid_r;
            product_valid_r <= dsp_product_valid_r;
            // The tile holds operands stable outside a valid issue. Sampling
            // every cycle keeps the DSP A/B input registers enabled locally;
            // input_valid_r still gates product retirement, so invalid cycles
            // cannot change any architectural result.
            packed_weight_r <= $signed({weight_hi, 9'b0, weight_lo});
            extended_activation_r <= $signed({10'b0, activation});
            unsigned_correction_r <= weight_hi[7]
                ? $signed({15'b0, activation, 25'b0})
                : 48'sd0;
            // Invalid products are discarded by the valid pipeline. Updating
            // the DSP and unpack registers every clock makes all DSP48E1
            // A/B/M/P clock enables local constants instead of a 64-DSP
            // control tree spanning the tensor tile.
            packed_product <= (packed_weight_r * extended_activation_r)
                + unsigned_correction_r;
            product_lo <= packed_product[15:0];
            product_hi <= packed_product[32:17];
        end
    end
endmodule

`default_nettype wire
