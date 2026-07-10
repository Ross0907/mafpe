`timescale 1ns / 1ps
`default_nettype none
`include "rtl/cifar10_sqizh_params.svh"

module cifar10_sqizh_readout_accelerator #(
    parameter int IMAGE_W = 32,
    parameter int IMAGE_H = 32,
    parameter int CHANNELS = 3,
    parameter int GRID = `CIFAR10_SQIZH_GRID,
    parameter int FEATURE_DIM = `CIFAR10_SQIZH_FEATURE_DIM,
    parameter int NUM_CLASSES = 10,
    parameter int NUM_STEPS = `CIFAR10_SQIZH_NUM_STEPS,
    parameter int INPUT_SHIFT = `CIFAR10_SQIZH_INPUT_SHIFT,
    parameter int SPIKE_SCORE_Q8 = `CIFAR10_SQIZH_SPIKE_SCORE_Q8,
    parameter int SQIZH_LATENCY = 34,
    parameter int ACC_W = 48
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire event_valid,
    output logic event_ready,
    input  wire [4:0] event_x,
    input  wire [4:0] event_y,
    input  wire [1:0] event_channel,
    input  wire [7:0] event_value,
    input  wire frame_done,

    output logic busy,
    output logic result_valid,
    output logic [3:0] predicted_label,
    output logic signed [31:0] predicted_score,

    output logic [5:0] debug_state,
    output logic [7:0] debug_class,
    output logic [15:0] debug_feature,
    output logic signed [31:0] debug_current0,
    output logic signed [31:0] debug_current1,
    output logic [7:0] debug_spikes0,
    output logic [7:0] debug_spikes1
);
    localparam int CELL = IMAGE_W / GRID;
    localparam int CELL_SHIFT = $clog2(CELL);
    localparam int GRID_CELLS = GRID * GRID;
    localparam signed [15:0] V_RESET_Q8 = -16'sd16640;
    localparam signed [15:0] U_RESET_Q8 = -16'sd3380;

    typedef enum logic [5:0] {
        ST_IDLE          = 6'd0,
        ST_CLEAR         = 6'd1,
        ST_STREAM        = 6'd2,
        ST_CURRENT       = 6'd3,
        ST_PREP_STEP     = 6'd4,
        ST_SQIZH_WAIT    = 6'd5,
        ST_CAPTURE_STEP  = 6'd6,
        ST_SCORE_INIT    = 6'd7,
        ST_SCORE_SCAN    = 6'd8,
        ST_RESULT        = 6'd9
    } state_t;

    state_t state;

    logic signed [15:0] weight_mem [0:NUM_CLASSES*FEATURE_DIM-1];
    logic signed [15:0] bias_mem [0:NUM_CLASSES-1];
    logic signed [ACC_W-1:0] accum [0:NUM_CLASSES-1];
    logic signed [15:0] class_current [0:NUM_CLASSES-1];
    logic signed [15:0] v_state [0:NUM_CLASSES-1];
    logic signed [15:0] u_state [0:NUM_CLASSES-1];
    logic signed [15:0] sqizh_v_in [0:NUM_CLASSES-1];
    logic signed [15:0] sqizh_u_in [0:NUM_CLASSES-1];
    logic signed [15:0] sqizh_i_in [0:NUM_CLASSES-1];
    logic signed [15:0] sqizh_v_out [0:NUM_CLASSES-1];
    logic signed [15:0] sqizh_u_out [0:NUM_CLASSES-1];
    logic sqizh_spike [0:NUM_CLASSES-1];
    logic [7:0] spike_count [0:NUM_CLASSES-1];

    logic [$clog2(NUM_CLASSES)-1:0] class_idx;
    logic [$clog2(NUM_CLASSES+1)-1:0] clear_idx;
    logic [$clog2(NUM_STEPS+1)-1:0] step_idx;
    logic [$clog2(SQIZH_LATENCY+1)-1:0] wait_count;
    logic [15:0] feature_idx;
    logic signed [31:0] score_w;

    assign event_ready = (state == ST_STREAM);
    assign busy = (state != ST_IDLE);
    assign debug_state = state;
    assign debug_class = {4'd0, class_idx};
    assign debug_feature = feature_idx;
    assign debug_current0 = {{16{class_current[0][15]}}, class_current[0]};
    assign debug_current1 = {{16{class_current[1][15]}}, class_current[1]};
    assign debug_spikes0 = spike_count[0];
    assign debug_spikes1 = spike_count[1];

    initial begin
        $readmemh("rtl/cifar10_sqizh_weights.mem", weight_mem);
        $readmemh("rtl/cifar10_sqizh_bias.mem", bias_mem);
    end

    function automatic logic signed [15:0] clamp_i16(input logic signed [ACC_W-1:0] value);
        begin
            if (value > 32767) begin
                clamp_i16 = 16'sd32767;
            end else if (value < -32768) begin
                clamp_i16 = -16'sd32768;
            end else begin
                clamp_i16 = value[15:0];
            end
        end
    endfunction

    function automatic logic signed [31:0] class_score(
        input logic signed [15:0] current_q8,
        input logic [7:0] spikes
    );
        begin
            class_score = {{16{current_q8[15]}}, current_q8} + $signed({24'd0, spikes}) * SPIKE_SCORE_Q8;
        end
    endfunction

    always_comb begin
        feature_idx = ({14'd0, event_channel} * GRID_CELLS[15:0])
                    + ({11'd0, (event_y >> CELL_SHIFT)} * GRID[15:0])
                    + {11'd0, (event_x >> CELL_SHIFT)};
    end

    for (genvar c = 0; c < NUM_CLASSES; c++) begin : gen_sqizh
        sqizh_neuron u_sqizh (
            .clk(clk),
            .v_in(sqizh_v_in[c]),
            .u_in(sqizh_u_in[c]),
            .i_in(sqizh_i_in[c]),
            .v_out(sqizh_v_out[c]),
            .u_out(sqizh_u_out[c]),
            .spike(sqizh_spike[c])
        );
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            result_valid <= 1'b0;
            predicted_label <= 4'd0;
            predicted_score <= '0;
            class_idx <= '0;
            clear_idx <= '0;
            step_idx <= '0;
            wait_count <= '0;
            score_w <= '0;
            for (int c = 0; c < NUM_CLASSES; c++) begin
                accum[c] <= '0;
                class_current[c] <= '0;
                v_state[c] <= V_RESET_Q8;
                u_state[c] <= U_RESET_Q8;
                sqizh_v_in[c] <= V_RESET_Q8;
                sqizh_u_in[c] <= U_RESET_Q8;
                sqizh_i_in[c] <= 16'sd0;
                spike_count[c] <= 8'd0;
            end
        end else begin
            result_valid <= 1'b0;
            unique case (state)
                ST_IDLE: begin
                    if (start) begin
                        clear_idx <= '0;
                        state <= ST_CLEAR;
                    end
                end

                ST_CLEAR: begin
                    accum[clear_idx] <= '0;
                    class_current[clear_idx] <= '0;
                    v_state[clear_idx] <= V_RESET_Q8;
                    u_state[clear_idx] <= U_RESET_Q8;
                    sqizh_v_in[clear_idx] <= V_RESET_Q8;
                    sqizh_u_in[clear_idx] <= U_RESET_Q8;
                    sqizh_i_in[clear_idx] <= 16'sd0;
                    spike_count[clear_idx] <= 8'd0;
                    if (clear_idx == NUM_CLASSES - 1) begin
                        state <= ST_STREAM;
                    end else begin
                        clear_idx <= clear_idx + 1'b1;
                    end
                end

                ST_STREAM: begin
                    if (event_valid && event_ready && event_x < IMAGE_W && event_y < IMAGE_H && event_channel < CHANNELS && feature_idx < FEATURE_DIM) begin
                        for (int c = 0; c < NUM_CLASSES; c++) begin
                            accum[c] <= accum[c] + ($signed({1'b0, event_value}) * weight_mem[c * FEATURE_DIM + feature_idx]);
                        end
                    end
                    if (frame_done) begin
                        class_idx <= '0;
                        state <= ST_CURRENT;
                    end
                end

                ST_CURRENT: begin
                    class_current[class_idx] <= clamp_i16({{(ACC_W-16){bias_mem[class_idx][15]}}, bias_mem[class_idx]} + (accum[class_idx] >>> INPUT_SHIFT));
                    if (class_idx == NUM_CLASSES - 1) begin
                        class_idx <= '0;
                        step_idx <= '0;
                        state <= ST_PREP_STEP;
                    end else begin
                        class_idx <= class_idx + 1'b1;
                    end
                end

                ST_PREP_STEP: begin
                    for (int c = 0; c < NUM_CLASSES; c++) begin
                        sqizh_v_in[c] <= v_state[c];
                        sqizh_u_in[c] <= u_state[c];
                        sqizh_i_in[c] <= class_current[c];
                    end
                    wait_count <= '0;
                    state <= ST_SQIZH_WAIT;
                end

                ST_SQIZH_WAIT: begin
                    if (wait_count == $clog2(SQIZH_LATENCY+1)'(SQIZH_LATENCY - 1)) begin
                        state <= ST_CAPTURE_STEP;
                    end else begin
                        wait_count <= wait_count + 1'b1;
                    end
                end

                ST_CAPTURE_STEP: begin
                    for (int c = 0; c < NUM_CLASSES; c++) begin
                        v_state[c] <= sqizh_v_out[c];
                        u_state[c] <= sqizh_u_out[c];
                        spike_count[c] <= spike_count[c] + {7'd0, sqizh_spike[c]};
                    end
                    if (step_idx == NUM_STEPS - 1) begin
                        class_idx <= '0;
                        predicted_label <= 4'd0;
                        predicted_score <= class_score(class_current[0], spike_count[0] + {7'd0, sqizh_spike[0]});
                        state <= ST_SCORE_INIT;
                    end else begin
                        step_idx <= step_idx + 1'b1;
                        state <= ST_PREP_STEP;
                    end
                end

                ST_SCORE_INIT: begin
                    class_idx <= 1;
                    state <= ST_SCORE_SCAN;
                end

                ST_SCORE_SCAN: begin
                    score_w = class_score(class_current[class_idx], spike_count[class_idx]);
                    if (score_w > predicted_score) begin
                        predicted_score <= score_w;
                        predicted_label <= {1'b0, class_idx};
                    end
                    if (class_idx == NUM_CLASSES - 1) begin
                        state <= ST_RESULT;
                    end else begin
                        class_idx <= class_idx + 1'b1;
                    end
                end

                ST_RESULT: begin
                    result_valid <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
