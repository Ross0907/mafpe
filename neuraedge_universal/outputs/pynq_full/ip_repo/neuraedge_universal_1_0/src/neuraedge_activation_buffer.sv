`timescale 1ns / 1ps
`default_nettype none

module neuraedge_activation_buffer #(
    parameter int DEPTH = 2048,
    parameter int ADDRESS_BITS = 11
) (
    input  wire clk,
    input  wire write_enable,
    input  wire [ADDRESS_BITS-1:0] write_address,
    input  wire [63:0] write_data,
    input  wire [ADDRESS_BITS-1:0] read_address,
    output logic [63:0] read_data
);
    (* ram_style = "block" *) logic [63:0] memory [0:DEPTH-1];
    logic [63:0] memory_read_data_r;

    always_ff @(posedge clk) begin
        if (write_enable) memory[write_address] <= write_data;
        memory_read_data_r <= memory[read_address];
        read_data <= memory_read_data_r;
    end

    initial begin
        if (DEPTH != (1 << ADDRESS_BITS)) $fatal(1, "activation buffer depth/address mismatch");
    end
endmodule

`default_nettype wire
