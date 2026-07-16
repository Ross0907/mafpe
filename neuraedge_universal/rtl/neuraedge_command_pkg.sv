`timescale 1ns / 1ps

package neuraedge_command_pkg;
    localparam logic [3:0] CMD_TRAIN_LINEAR_BACKWARD = 4'h0;
    localparam logic [3:0] CMD_LOAD_ACTIVATIONS = 4'h1;
    localparam logic [3:0] CMD_LOAD_WEIGHTS_OUTER = 4'h2;
    localparam logic [3:0] CMD_LOAD_WEIGHTS_DOT8 = 4'h3;
    localparam logic [3:0] CMD_COMPUTE_DENSE_OUTER = 4'h4;
    localparam logic [3:0] CMD_COMPUTE_DENSE_DOT8 = 4'h5;
    localparam logic [3:0] CMD_COMPUTE_BINARY = 4'h6;
    localparam logic [3:0] CMD_COMPUTE_TERNARY = 4'h7;
    localparam logic [3:0] CMD_ALPHA_XNOR = 4'h8;
    localparam logic [3:0] CMD_VECTOR_ALU = 4'h9;
    localparam logic [3:0] CMD_LAYERNORM = 4'ha;
    localparam logic [3:0] CMD_LOG_SOFTMAX = 4'hb;
    localparam logic [3:0] CMD_LIF_STEP = 4'hc;
    localparam logic [3:0] CMD_SGD_UPDATE = 4'hd;
    localparam logic [3:0] CMD_ACTIVATION_LUT = 4'he;
    localparam logic [3:0] CMD_CLEAR_STATUS = 4'hf;

    localparam logic [1:0] MODE_DENSE_OUTER = 2'd0;
    localparam logic [1:0] MODE_DENSE_DOT8 = 2'd1;
    localparam logic [1:0] MODE_BINARY_WEIGHTED = 2'd2;
    localparam logic [1:0] MODE_TERNARY_WEIGHTED = 2'd3;

    // The 128-bit command word is deliberately fixed-width so software can
    // submit it as four ordinary 32-bit AXI-Lite register writes.
    // [3:0]   opcode
    // [7:4]   active output rows
    // [13:8]  active output columns
    // [25:14] reduction issue count
    // [33:26] activation zero point
    // [41:34] weight zero point
    // [127:42] tensor commands: reserved and required to be zero
    // Other operators reinterpret the payload as documented in
    // docs/stream_protocol.md; opcode placement never changes.
endpackage
