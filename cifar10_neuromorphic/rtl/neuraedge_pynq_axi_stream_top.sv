`timescale 1ns / 1ps
`default_nettype none

module neuraedge_pynq_axi_stream_top #(
    parameter int EVENT_W = 24
) (
    input  wire ap_clk,
    input  wire ap_rst_n,

    input  wire s_axis_event_tvalid,
    output logic s_axis_event_tready,
    input  wire [EVENT_W-1:0] s_axis_event_tdata,
    input  wire s_axis_event_tlast,

    output logic m_axis_result_tvalid,
    input  wire m_axis_result_tready,
    output logic [31:0] m_axis_result_tdata,

    output logic busy
);
    typedef enum logic [2:0] {
        WR_IDLE,
        WR_WAIT_READY,
        WR_SEND_EVENT,
        WR_STREAM,
        WR_WAIT_FRAME,
        WR_SEND_FRAME,
        WR_RESULT
    } wrap_state_t;

    wrap_state_t wrap_state;

    logic core_start;
    logic core_event_valid;
    logic core_event_ready;
    logic [4:0] core_event_x;
    logic [4:0] core_event_y;
    logic [1:0] core_event_channel;
    logic [7:0] core_event_value;
    logic core_frame_done;
    logic core_result_valid;
    logic [EVENT_W-1:0] buffered_event;
    logic buffered_last;
    logic [3:0] core_predicted_label;
    logic signed [31:0] core_predicted_score;
    logic [5:0] unused_debug_state;
    logic [7:0] unused_debug_token;
    logic [7:0] unused_debug_key;
    logic [7:0] unused_debug_dim;
    logic signed [31:0] unused_debug_dot;
    logic signed [31:0] unused_debug_cls0;
    logic signed [31:0] unused_debug_cls1;
    logic [31:0] unused_active_tokens;
    logic [31:0] unused_skipped_tokens;
    logic [31:0] unused_qkv_macs;
    logic [31:0] unused_event_count;
    logic [31:0] unused_attention_macs;
    logic [31:0] unused_ffn_macs;
    logic [31:0] unused_head_macs;
    logic [31:0] unused_sqizh_steps;

    assign s_axis_event_tready = (wrap_state == WR_IDLE) || ((wrap_state == WR_STREAM) && core_event_ready);

    neuraedge_sqizh_transformer_accelerator u_accelerator (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(core_start),
        .event_valid(core_event_valid),
        .event_ready(core_event_ready),
        .event_x(core_event_x),
        .event_y(core_event_y),
        .event_channel(core_event_channel),
        .event_value(core_event_value),
        .frame_done(core_frame_done),
        .busy(busy),
        .result_valid(core_result_valid),
        .predicted_label(core_predicted_label),
        .predicted_score(core_predicted_score),
        .debug_state(unused_debug_state),
        .debug_token(unused_debug_token),
        .debug_key(unused_debug_key),
        .debug_dim(unused_debug_dim),
        .debug_dot(unused_debug_dot),
        .debug_cls0(unused_debug_cls0),
        .debug_cls1(unused_debug_cls1),
        .debug_event_count(unused_event_count),
        .debug_active_tokens(unused_active_tokens),
        .debug_skipped_tokens(unused_skipped_tokens),
        .debug_qkv_macs(unused_qkv_macs),
        .debug_attention_macs(unused_attention_macs),
        .debug_ffn_macs(unused_ffn_macs),
        .debug_head_macs(unused_head_macs),
        .debug_sqizh_steps(unused_sqizh_steps)
    );

    always_ff @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            wrap_state <= WR_IDLE;
            core_start <= 1'b0;
            core_event_valid <= 1'b0;
            core_event_x <= 5'd0;
            core_event_y <= 5'd0;
            core_event_channel <= 2'd0;
            core_event_value <= 8'd0;
            core_frame_done <= 1'b0;
            buffered_event <= '0;
            buffered_last <= 1'b0;
            m_axis_result_tvalid <= 1'b0;
            m_axis_result_tdata <= 32'd0;
        end else begin
            core_start <= 1'b0;
            core_event_valid <= 1'b0;
            core_frame_done <= 1'b0;

            unique case (wrap_state)
                WR_IDLE: begin
                    if (s_axis_event_tvalid && s_axis_event_tready) begin
                        buffered_event <= s_axis_event_tdata;
                        buffered_last <= s_axis_event_tlast;
                        core_start <= 1'b1;
                        wrap_state <= WR_WAIT_READY;
                    end
                end

                WR_WAIT_READY: begin
                    if (core_event_ready) begin
                        core_event_channel <= buffered_event[19:18];
                        core_event_x <= buffered_event[17:13];
                        core_event_y <= buffered_event[12:8];
                        core_event_value <= buffered_event[7:0];
                        core_event_valid <= 1'b1;
                        wrap_state <= WR_SEND_EVENT;
                    end
                end

                WR_SEND_EVENT: begin
                    if (buffered_last) begin
                        wrap_state <= WR_WAIT_FRAME;
                    end else begin
                        wrap_state <= WR_STREAM;
                    end
                end

                WR_STREAM: begin
                    if (s_axis_event_tvalid && s_axis_event_tready) begin
                        buffered_last <= s_axis_event_tlast;
                        core_event_channel <= s_axis_event_tdata[19:18];
                        core_event_x <= s_axis_event_tdata[17:13];
                        core_event_y <= s_axis_event_tdata[12:8];
                        core_event_value <= s_axis_event_tdata[7:0];
                        core_event_valid <= 1'b1;
                        wrap_state <= WR_SEND_EVENT;
                    end
                end

                WR_WAIT_FRAME: begin
                    if (core_event_ready) begin
                        core_frame_done <= 1'b1;
                        wrap_state <= WR_SEND_FRAME;
                    end
                end

                WR_SEND_FRAME: begin
                    wrap_state <= WR_RESULT;
                end

                WR_RESULT: begin
                    if (core_result_valid) begin
                        m_axis_result_tdata <= {core_predicted_score[27:0], core_predicted_label};
                        m_axis_result_tvalid <= 1'b1;
                    end
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
