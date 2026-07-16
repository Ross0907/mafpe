`timescale 1ns / 1ps
`default_nettype none

module neuraedge_weight_bank #(
    parameter int DEPTH = 256,
    parameter int ADDRESS_BITS = 8
) (
    input  wire clk,
    input  wire write_enable,
    input  wire [ADDRESS_BITS-1:0] write_address,
    input  wire [1:0] write_segment,
    input  wire [63:0] write_data,
    input  wire [ADDRESS_BITS-1:0] read_address,
    output logic [127:0] read_data
);
    // Four independent 64-bit memories map directly to BRAM primitives. The
    // segment write enable avoids the 3D partial-write pattern that Vivado
    // otherwise expands into half a million flip-flops.
    (* ram_style = "block" *) logic [63:0] memory0 [0:DEPTH-1];
    (* ram_style = "block" *) logic [63:0] memory1 [0:DEPTH-1];
    logic [127:0] memory_read_data_r;

    always_ff @(posedge clk) begin
        if (write_enable) begin
            unique case (write_segment)
                2'd0: memory0[write_address] <= write_data;
                2'd1: memory1[write_address] <= write_data;
                default: memory1[write_address] <= write_data;
            endcase
        end
        memory_read_data_r[63:0] <= memory0[read_address];
        memory_read_data_r[127:64] <= memory1[read_address];
        read_data <= memory_read_data_r;
    end

    initial begin
        if (DEPTH != (1 << ADDRESS_BITS)) $fatal(1, "weight bank depth/address mismatch");
    end
endmodule

`default_nettype wire
