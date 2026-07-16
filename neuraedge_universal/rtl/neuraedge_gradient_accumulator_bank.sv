`timescale 1ns / 1ps
`default_nettype none

// Stores one lane of exact 47-bit backward-gradient partial sums.
module neuraedge_gradient_accumulator_bank (
    input  wire clk,
    input  wire write_enable,
    input  wire [3:0] write_address,
    input  wire signed [46:0] write_data,
    input  wire [3:0] read_address,
    output wire signed [46:0] read_data
);
    // One asynchronous read and one synchronous write match a 7-series
    // distributed-RAM inference template. There is intentionally no reset:
    // the parent selects zero until a complete non-accumulating command has
    // established valid contents for every active address.
    (* ram_style = "distributed" *)
    logic signed [46:0] memory [0:15];

    assign read_data = memory[read_address];

    always_ff @(posedge clk) begin
        if (write_enable) memory[write_address] <= write_data;
    end
endmodule

`default_nettype wire
