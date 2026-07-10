`timescale 1ns / 1ps
`default_nettype none
`include "rtl/cifar10_spikingformer_head_params.svh"
`ifndef CIFAR10_SF_HEAD_INFER_SKIP_ABS_THRESH
`define CIFAR10_SF_HEAD_INFER_SKIP_ABS_THRESH 2048
`endif

module cifar10_spikingformer_head_accelerator #(
    parameter int FEATURE_DIM = `CIFAR10_SF_HEAD_FEATURE_DIM,
    parameter int NUM_CLASSES = `CIFAR10_SF_HEAD_NUM_CLASSES,
    parameter int FEATURE_BITS = `CIFAR10_SF_HEAD_FEATURE_BITS,
    parameter int WEIGHT_BITS = `CIFAR10_SF_HEAD_WEIGHT_BITS,
    parameter int ACC_BITS = `CIFAR10_SF_HEAD_ACC_BITS,
    parameter int ENABLE_ONCHIP_LEARNING = 1,
    parameter int INFER_SKIP_ABS_THRESH = `CIFAR10_SF_HEAD_INFER_SKIP_ABS_THRESH,
    parameter int LEARN_SHIFT = 14,
    parameter int BIAS_LEARN_DELTA = 1,
    parameter logic signed [ACC_BITS-1:0] LEARN_MARGIN_Q = 64'sd268435456
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,
    input  wire feature_valid,
    output logic feature_ready,
    input  wire signed [FEATURE_BITS-1:0] feature_value,
    input  wire frame_done,
    input  wire train_enable,
    input  wire [3:0] train_label,

    output logic busy,
    output logic result_valid,
    output logic [3:0] predicted_label,
    output logic signed [ACC_BITS-1:0] predicted_score,
    output logic train_update_valid,
    output logic train_update_applied,

    output logic [5:0] debug_state,
    output logic [15:0] debug_feature,
    output logic [7:0] debug_class,
    output logic [31:0] debug_update_count,
    output logic [31:0] debug_infer_skip_count,
    output logic [31:0] debug_train_skip_count
);
    localparam int FEATURE_W = $clog2(FEATURE_DIM + 1);
    localparam int CLASS_W = $clog2(NUM_CLASSES + 1);

    typedef enum logic [5:0] {
        ST_IDLE       = 6'd0,
        ST_LOAD       = 6'd1,
        ST_CLASS_INIT = 6'd2,
        ST_MAC_READ   = 6'd3,
        ST_MAC_MUL    = 6'd4,
        ST_MAC_ACCUM  = 6'd5,
        ST_CLASS_DONE = 6'd6,
        ST_RESULT     = 6'd7,
        ST_TRAIN_CHECK      = 6'd8,
        ST_TRAIN_READ_TRUE  = 6'd9,
        ST_TRAIN_CALC_TRUE  = 6'd10,
        ST_TRAIN_WRITE_TRUE = 6'd11,
        ST_TRAIN_READ_PRED  = 6'd12,
        ST_TRAIN_CALC_PRED  = 6'd13,
        ST_TRAIN_WRITE_PRED = 6'd14,
        ST_TRAIN_BIAS_TRUE  = 6'd15,
        ST_TRAIN_BIAS_PRED  = 6'd16
    } state_t;

    state_t state;

    (* ram_style = "distributed" *) logic signed [FEATURE_BITS-1:0] feature_mem [0:FEATURE_DIM-1];
    (* ram_style = "block" *) logic signed [WEIGHT_BITS-1:0] weight_mem [0:NUM_CLASSES*FEATURE_DIM-1];
    logic signed [ACC_BITS-1:0] bias_mem [0:NUM_CLASSES-1];
    logic signed [ACC_BITS-1:0] class_score_mem [0:NUM_CLASSES-1];

    logic [FEATURE_W-1:0] feature_idx;
    logic [CLASS_W-1:0] class_idx;
    logic signed [ACC_BITS-1:0] accum;
    logic signed [FEATURE_BITS-1:0] feature_eff_r;
    logic signed [WEIGHT_BITS-1:0] weight_r;
    logic signed [ACC_BITS-1:0] product_r;
    logic signed [WEIGHT_BITS-1:0] train_true_weight_r;
    logic signed [WEIGHT_BITS-1:0] train_pred_weight_r;
    logic signed [WEIGHT_BITS-1:0] train_write_weight_r;
    logic [3:0] train_label_r;
    logic [3:0] train_pred_label_r;
    logic [31:0] update_count;
    logic [31:0] infer_skip_count;
    logic [31:0] train_skip_count;

    assign feature_ready = (state == ST_LOAD);
    assign busy = (state != ST_IDLE);
    assign debug_state = state;
    assign debug_feature = {{(16-FEATURE_W){1'b0}}, feature_idx};
    assign debug_class = {4'd0, 4'(class_idx)};
    assign debug_update_count = update_count;
    assign debug_infer_skip_count = infer_skip_count;
    assign debug_train_skip_count = train_skip_count;

    initial begin
        $readmemh("rtl/cifar10_spikingformer_head_weights.mem", weight_mem);
        $readmemh("rtl/cifar10_spikingformer_head_bias.mem", bias_mem);
    end

    function automatic logic signed [WEIGHT_BITS-1:0] sat_weight(
        input logic signed [ACC_BITS-1:0] value
    );
        begin
            if (value > ((1 <<< (WEIGHT_BITS - 1)) - 1)) begin
                sat_weight = {1'b0, {(WEIGHT_BITS-1){1'b1}}};
            end else if (value < -(1 <<< (WEIGHT_BITS - 1))) begin
                sat_weight = {1'b1, {(WEIGHT_BITS-1){1'b0}}};
            end else begin
                sat_weight = value[WEIGHT_BITS-1:0];
            end
        end
    endfunction

    function automatic int weight_addr(input int cls, input int feat);
        begin
            weight_addr = cls * FEATURE_DIM + feat;
        end
    endfunction

    function automatic logic signed [ACC_BITS-1:0] learn_delta(
        input logic signed [FEATURE_BITS-1:0] value
    );
        logic signed [ACC_BITS-1:0] extended;
        begin
            extended = {{(ACC_BITS-FEATURE_BITS){value[FEATURE_BITS-1]}}, value};
            learn_delta = extended >>> LEARN_SHIFT;
        end
    endfunction

    function automatic logic [FEATURE_BITS:0] abs_feature(
        input logic signed [FEATURE_BITS-1:0] value
    );
        logic signed [FEATURE_BITS:0] extended;
        begin
            extended = {value[FEATURE_BITS-1], value};
            if (extended < 0) begin
                abs_feature = (0 - extended);
            end else begin
                abs_feature = extended[FEATURE_BITS:0];
            end
        end
    endfunction

    function automatic logic skip_feature(
        input logic signed [FEATURE_BITS-1:0] value
    );
        begin
            if (INFER_SKIP_ABS_THRESH <= 0) begin
                skip_feature = 1'b0;
            end else begin
                skip_feature = abs_feature(value) <= INFER_SKIP_ABS_THRESH;
            end
        end
    endfunction

    function automatic logic signed [FEATURE_BITS-1:0] effective_feature(
        input logic signed [FEATURE_BITS-1:0] value
    );
        begin
            if (skip_feature(value)) begin
                effective_feature = '0;
            end else begin
                effective_feature = value;
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            feature_idx <= '0;
            class_idx <= '0;
            accum <= '0;
            product_r <= '0;
            result_valid <= 1'b0;
            train_update_valid <= 1'b0;
            train_update_applied <= 1'b0;
            predicted_label <= 4'd0;
            predicted_score <= '0;
            train_write_weight_r <= '0;
            train_label_r <= 4'd0;
            train_pred_label_r <= 4'd0;
            update_count <= 32'd0;
            infer_skip_count <= 32'd0;
            train_skip_count <= 32'd0;
            for (int c = 0; c < NUM_CLASSES; c++) begin
                class_score_mem[c] <= '0;
            end
        end else begin
            result_valid <= 1'b0;
            train_update_valid <= 1'b0;
            train_update_applied <= 1'b0;
            unique case (state)
                ST_IDLE: begin
                    if (start) begin
                        feature_idx <= '0;
                        state <= ST_LOAD;
                    end
                end

                ST_LOAD: begin
                    if (feature_valid && feature_ready) begin
                        if (feature_idx == FEATURE_DIM - 1) begin
                            feature_idx <= '0;
                            class_idx <= '0;
                            state <= ST_CLASS_INIT;
                        end else begin
                            feature_idx <= feature_idx + 1'b1;
                        end
                    end else if (frame_done && feature_idx == FEATURE_DIM) begin
                        feature_idx <= '0;
                        class_idx <= '0;
                        state <= ST_CLASS_INIT;
                    end
                end

                ST_CLASS_INIT: begin
                    accum <= bias_mem[class_idx];
                    feature_idx <= '0;
                    state <= ST_MAC_READ;
                end

                ST_MAC_READ: begin
                    if (skip_feature(feature_mem[feature_idx])) begin
                        infer_skip_count <= infer_skip_count + 1'b1;
                        state <= ST_CLASS_DONE;
                    end else begin
                        state <= ST_MAC_MUL;
                    end
                end

                ST_MAC_MUL: begin
                    product_r <= $signed(feature_eff_r) * $signed(weight_r);
                    state <= ST_MAC_ACCUM;
                end

                ST_MAC_ACCUM: begin
                    accum <= accum + product_r;
                    state <= ST_CLASS_DONE;
                end

                ST_CLASS_DONE: begin
                    if (feature_idx == FEATURE_DIM - 1) begin
                        class_score_mem[class_idx] <= accum;
                        if (class_idx == 0) begin
                            predicted_label <= 4'd0;
                            predicted_score <= accum;
                        end else if (accum > predicted_score) begin
                            predicted_label <= 4'(class_idx);
                            predicted_score <= accum;
                        end
                        if (class_idx == NUM_CLASSES - 1) begin
                            if (ENABLE_ONCHIP_LEARNING != 0) begin
                                state <= ST_TRAIN_CHECK;
                            end else begin
                                state <= ST_RESULT;
                            end
                        end else begin
                            class_idx <= class_idx + 1'b1;
                            state <= ST_CLASS_INIT;
                        end
                    end else begin
                        feature_idx <= feature_idx + 1'b1;
                        state <= ST_MAC_READ;
                    end
                end

                ST_TRAIN_CHECK: begin
                    train_update_valid <= 1'b1;
                    if (train_enable
                            && train_label < NUM_CLASSES
                            && train_label != predicted_label
                            && (predicted_score - class_score_mem[train_label]) <= LEARN_MARGIN_Q) begin
                        feature_idx <= '0;
                        train_label_r <= train_label;
                        train_pred_label_r <= predicted_label;
                        train_update_applied <= 1'b1;
                        update_count <= update_count + 1'b1;
                        state <= ST_TRAIN_READ_TRUE;
                    end else begin
                        state <= ST_RESULT;
                    end
                end

                ST_TRAIN_READ_TRUE: begin
                    if (skip_feature(feature_mem[feature_idx])) begin
                        train_skip_count <= train_skip_count + 1'b1;
                        if (feature_idx == FEATURE_DIM - 1) begin
                            state <= ST_TRAIN_BIAS_TRUE;
                        end else begin
                            feature_idx <= feature_idx + 1'b1;
                            state <= ST_TRAIN_READ_TRUE;
                        end
                    end else begin
                        state <= ST_TRAIN_CALC_TRUE;
                    end
                end

                ST_TRAIN_CALC_TRUE: begin
                    train_write_weight_r <= sat_weight(
                        {{(ACC_BITS-WEIGHT_BITS){train_true_weight_r[WEIGHT_BITS-1]}}, train_true_weight_r}
                        + learn_delta(feature_eff_r)
                    );
                    state <= ST_TRAIN_WRITE_TRUE;
                end

                ST_TRAIN_WRITE_TRUE: begin
                    state <= ST_TRAIN_READ_PRED;
                end

                ST_TRAIN_READ_PRED: begin
                    state <= ST_TRAIN_CALC_PRED;
                end

                ST_TRAIN_CALC_PRED: begin
                    train_write_weight_r <= sat_weight(
                        {{(ACC_BITS-WEIGHT_BITS){train_pred_weight_r[WEIGHT_BITS-1]}}, train_pred_weight_r}
                        - learn_delta(feature_eff_r)
                    );
                    state <= ST_TRAIN_WRITE_PRED;
                end

                ST_TRAIN_WRITE_PRED: begin
                    if (feature_idx == FEATURE_DIM - 1) begin
                        state <= ST_TRAIN_BIAS_TRUE;
                    end else begin
                        feature_idx <= feature_idx + 1'b1;
                        state <= ST_TRAIN_READ_TRUE;
                    end
                end

                ST_TRAIN_BIAS_TRUE: begin
                    state <= ST_TRAIN_BIAS_PRED;
                end

                ST_TRAIN_BIAS_PRED: begin
                    state <= ST_RESULT;
                end

                ST_RESULT: begin
                    result_valid <= 1'b1;
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

    always_ff @(posedge clk) begin
        unique case (state)
            ST_LOAD: begin
                if (feature_valid && feature_ready) begin
                    feature_mem[feature_idx] <= feature_value;
                end
            end

            ST_MAC_READ: begin
                feature_eff_r <= effective_feature(feature_mem[feature_idx]);
                if (!skip_feature(feature_mem[feature_idx])) begin
                    weight_r <= weight_mem[weight_addr(int'(class_idx), int'(feature_idx))];
                end
            end

            ST_TRAIN_READ_TRUE: begin
                feature_eff_r <= effective_feature(feature_mem[feature_idx]);
                if (!skip_feature(feature_mem[feature_idx])) begin
                    train_true_weight_r <= weight_mem[weight_addr(int'(train_label_r), int'(feature_idx))];
                end
            end

            ST_TRAIN_WRITE_TRUE: begin
                weight_mem[weight_addr(int'(train_label_r), int'(feature_idx))] <= train_write_weight_r;
            end

            ST_TRAIN_READ_PRED: begin
                train_pred_weight_r <= weight_mem[weight_addr(int'(train_pred_label_r), int'(feature_idx))];
            end

            ST_TRAIN_WRITE_PRED: begin
                weight_mem[weight_addr(int'(train_pred_label_r), int'(feature_idx))] <= train_write_weight_r;
            end

            ST_TRAIN_BIAS_TRUE: begin
                bias_mem[train_label_r] <= bias_mem[train_label_r] + BIAS_LEARN_DELTA;
            end

            ST_TRAIN_BIAS_PRED: begin
                bias_mem[train_pred_label_r] <= bias_mem[train_pred_label_r] - BIAS_LEARN_DELTA;
            end

            default: begin
            end
        endcase
    end
endmodule

`default_nettype wire
