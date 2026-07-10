`timescale 1ns / 1ps
`default_nettype none

module cifar10_spikingformer_head_pynq_axi_stream_top (
    input  wire ap_clk,
    input  wire ap_rst_n,

    input  wire s_axis_feature_tvalid,
    output logic s_axis_feature_tready,
    input  wire signed [15:0] s_axis_feature_tdata,
    input  wire s_axis_feature_tlast,

    input  wire train_enable,
    input  wire [3:0] train_label,

    output logic m_axis_result_tvalid,
    input  wire m_axis_result_tready,
    output logic [31:0] m_axis_result_tdata,

    output logic busy,
    output logic [31:0] debug_infer_skip_count,
    output logic [31:0] debug_train_skip_count
);
    typedef enum logic [2:0] {
        WR_IDLE        = 3'd0,
        WR_WAIT_READY  = 3'd1,
        WR_SEND_BUFFER = 3'd2,
        WR_STREAM      = 3'd3,
        WR_FRAME_DONE  = 3'd4,
        WR_WAIT_RESULT = 3'd5,
        WR_HOLD_RESULT = 3'd6
    } wrap_state_t;

    wrap_state_t wrap_state;

    logic core_start;
    logic core_feature_valid;
    logic core_feature_ready;
    logic signed [15:0] core_feature_value;
    logic core_frame_done;
    logic core_busy;
    logic core_result_valid;
    logic [3:0] core_predicted_label;
    logic signed [63:0] core_predicted_score;
    logic core_train_update_valid;
    logic core_train_update_applied;
    logic [5:0] unused_debug_state;
    logic [15:0] unused_debug_feature;
    logic [7:0] unused_debug_class;
    logic [31:0] core_update_count;
    logic [31:0] core_infer_skip_count;
    logic [31:0] core_train_skip_count;
    logic signed [15:0] buffered_feature;
    logic buffered_last;
    logic update_applied_latched;

    assign busy = core_busy || (wrap_state != WR_IDLE);
    assign s_axis_feature_tready = (wrap_state == WR_IDLE) || (wrap_state == WR_STREAM && core_feature_ready);

    cifar10_spikingformer_head_accelerator u_head (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(core_start),
        .feature_valid(core_feature_valid),
        .feature_ready(core_feature_ready),
        .feature_value(core_feature_value),
        .frame_done(core_frame_done),
        .train_enable(train_enable),
        .train_label(train_label),
        .busy(core_busy),
        .result_valid(core_result_valid),
        .predicted_label(core_predicted_label),
        .predicted_score(core_predicted_score),
        .train_update_valid(core_train_update_valid),
        .train_update_applied(core_train_update_applied),
        .debug_state(unused_debug_state),
        .debug_feature(unused_debug_feature),
        .debug_class(unused_debug_class),
        .debug_update_count(core_update_count),
        .debug_infer_skip_count(core_infer_skip_count),
        .debug_train_skip_count(core_train_skip_count)
    );

    assign debug_infer_skip_count = core_infer_skip_count;
    assign debug_train_skip_count = core_train_skip_count;

    always_ff @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            wrap_state <= WR_IDLE;
            core_start <= 1'b0;
            core_feature_valid <= 1'b0;
            core_feature_value <= '0;
            core_frame_done <= 1'b0;
            buffered_feature <= '0;
            buffered_last <= 1'b0;
            update_applied_latched <= 1'b0;
            m_axis_result_tvalid <= 1'b0;
            m_axis_result_tdata <= 32'd0;
        end else begin
            core_start <= 1'b0;
            core_feature_valid <= 1'b0;
            core_frame_done <= 1'b0;

            if (core_train_update_applied) begin
                update_applied_latched <= 1'b1;
            end

            unique case (wrap_state)
                WR_IDLE: begin
                    update_applied_latched <= 1'b0;
                    if (s_axis_feature_tvalid && s_axis_feature_tready) begin
                        buffered_feature <= s_axis_feature_tdata;
                        buffered_last <= s_axis_feature_tlast;
                        core_start <= 1'b1;
                        wrap_state <= WR_WAIT_READY;
                    end
                end

                WR_WAIT_READY: begin
                    if (core_feature_ready) begin
                        core_feature_value <= buffered_feature;
                        core_feature_valid <= 1'b1;
                        wrap_state <= WR_SEND_BUFFER;
                    end
                end

                WR_SEND_BUFFER: begin
                    if (buffered_last) begin
                        wrap_state <= WR_FRAME_DONE;
                    end else begin
                        wrap_state <= WR_STREAM;
                    end
                end

                WR_STREAM: begin
                    if (s_axis_feature_tvalid && s_axis_feature_tready) begin
                        core_feature_value <= s_axis_feature_tdata;
                        core_feature_valid <= 1'b1;
                        if (s_axis_feature_tlast) begin
                            wrap_state <= WR_FRAME_DONE;
                        end
                    end
                end

                WR_FRAME_DONE: begin
                    core_frame_done <= 1'b1;
                    wrap_state <= WR_WAIT_RESULT;
                end

                WR_WAIT_RESULT: begin
                    if (core_result_valid) begin
                        m_axis_result_tdata <= {
                            core_predicted_score[15:0],
                            core_update_count[10:0],
                            update_applied_latched,
                            core_predicted_label
                        };
                        m_axis_result_tvalid <= 1'b1;
                        wrap_state <= WR_HOLD_RESULT;
                    end
                end

                WR_HOLD_RESULT: begin
                    if (m_axis_result_tvalid && m_axis_result_tready) begin
                        m_axis_result_tvalid <= 1'b0;
                        wrap_state <= WR_IDLE;
                    end
                end

                default: begin
                    wrap_state <= WR_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
