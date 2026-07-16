`timescale 1ns / 1ps
`default_nettype none

// Simple-dual-port storage for the compacted list of nonzero activation tiles.
// Both ports are synchronous, matching the canonical Vivado BRAM template.
module neuraedge_active_index_buffer #(
    parameter int DEPTH = 2048,
    parameter int ADDRESS_BITS = 11
) (
    input  wire clk,
    input  wire write_enable,
    input  wire [ADDRESS_BITS-1:0] write_address,
    input  wire [ADDRESS_BITS-1:0] write_data,
    input  wire read_enable,
    input  wire [ADDRESS_BITS-1:0] read_address,
    output logic [ADDRESS_BITS-1:0] read_data
);
    (* ram_style = "block" *) logic [ADDRESS_BITS-1:0] memory [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (write_enable) memory[write_address] <= write_data;
        if (read_enable) read_data <= memory[read_address];
    end

    initial begin
        if (DEPTH != (1 << ADDRESS_BITS)) begin
            $fatal(1, "active-index depth must equal 2^ADDRESS_BITS");
        end
    end
endmodule

`default_nettype wire
