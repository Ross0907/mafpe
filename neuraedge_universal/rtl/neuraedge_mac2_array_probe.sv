`timescale 1ns / 1ps
`default_nettype none

// Synthesis-only probe used to prove packed-MAC density on the target FPGA.
module neuraedge_mac2_array_probe #(
    parameter int PAIRS = 128
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        step,
    output logic [31:0] checksum
);
    logic [31:0] stimulus [0:PAIRS-1];
    wire [15:0] product_lo [0:PAIRS-1];
    wire [15:0] product_hi [0:PAIRS-1];
    wire product_valid [0:PAIRS-1];

    for (genvar pair_idx = 0; pair_idx < PAIRS; pair_idx++) begin : g_pair
        neuraedge_mac2_u8 u_mac (
            .clk(clk),
            .rst_n(rst_n),
            .enable(step),
            .activation(stimulus[pair_idx][7:0]),
            .weight_lo(stimulus[pair_idx][15:8]),
            .weight_hi(stimulus[pair_idx][23:16]),
            .product_valid(product_valid[pair_idx]),
            .product_lo(product_lo[pair_idx]),
            .product_hi(product_hi[pair_idx])
        );

        always_ff @(posedge clk) begin
            if (!rst_n) begin
                stimulus[pair_idx] <= 32'h6d2b79f5 ^ (pair_idx * 32'h9e3779b9);
            end else if (step) begin
                stimulus[pair_idx] <= {stimulus[pair_idx][30:0],
                    stimulus[pair_idx][31] ^ stimulus[pair_idx][21]
                    ^ stimulus[pair_idx][1] ^ stimulus[pair_idx][0]};
            end
        end
    end

    always_comb begin
        checksum = 32'd0;
        for (int pair_idx = 0; pair_idx < PAIRS; pair_idx++) begin
            checksum = checksum ^ {product_hi[pair_idx], product_lo[pair_idx]};
        end
    end
endmodule

`default_nettype wire
