`timescale 1ns / 1ps
`default_nettype none

module neuraedge_sqizh_transformer_accelerator #(
    parameter int IMAGE_W = 32,
    parameter int IMAGE_H = 32,
    parameter int CHANNELS = 3,
    parameter int PATCH_SIZE = 4,
    parameter int PATCH_COLS = IMAGE_W / PATCH_SIZE,
    parameter int PATCH_ROWS = IMAGE_H / PATCH_SIZE,
    parameter int PATCH_TOKENS = PATCH_COLS * PATCH_ROWS,
    parameter int TOTAL_TOKENS = PATCH_TOKENS + 1,
    parameter int EMBED_DIM = 4,
    parameter int NUM_CLASSES = 10,
    parameter int NUM_STEPS = 8,
    parameter int OUT_STEPS = 4,
    parameter int SQIZH_LATENCY = 34,
    parameter int TOKEN_INPUT_SHIFT = 0,
    parameter int DOT_SHIFT = 3,
    parameter int ATTN_SHIFT = 4,
    parameter int FFN_SHIFT = 4,
    parameter int HEAD_SHIFT = 14,
    parameter int CLASS_CURRENT_SHIFT = 20,
    parameter int ACC_W = 32
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
    output logic [7:0] debug_token,
    output logic [7:0] debug_key,
    output logic [7:0] debug_dim,
    output logic signed [31:0] debug_dot,
    output logic signed [31:0] debug_cls0,
    output logic signed [31:0] debug_cls1,
    output logic [31:0] debug_event_count,
    output logic [31:0] debug_active_tokens,
    output logic [31:0] debug_skipped_tokens,
    output logic [31:0] debug_qkv_macs,
    output logic [31:0] debug_attention_macs,
    output logic [31:0] debug_ffn_macs,
    output logic [31:0] debug_head_macs,
    output logic [31:0] debug_sqizh_steps
);
    localparam int TOKEN_W = $clog2(TOTAL_TOKENS);
    localparam int PATCH_W = $clog2(PATCH_TOKENS);
    localparam int DIM_W = $clog2(EMBED_DIM);
    localparam int CLASS_W = $clog2(NUM_CLASSES);
    localparam int STEP_W = $clog2(NUM_STEPS + 1);
    localparam int OUT_STEP_W = $clog2(OUT_STEPS + 1);
    localparam int WAIT_W = $clog2(SQIZH_LATENCY + 1);
    localparam int CLEAR_COUNT = TOTAL_TOKENS * EMBED_DIM;
    localparam int CLEAR_W = $clog2(CLEAR_COUNT + 1);
    localparam logic signed [15:0] V_RESET_Q8 = -16'sd16640;
    localparam logic signed [15:0] U_RESET_Q8 = -16'sd3380;
    localparam logic signed [15:0] TOKEN_BASE_Q8 = 16'sd14848;
    localparam logic signed [15:0] CLASS_BASE_Q8 = 16'sd10240;

    typedef enum logic [5:0] {
        ST_IDLE          = 6'd0,
        ST_CLEAR         = 6'd1,
        ST_STREAM        = 6'd2,
        ST_EVENT_CALC    = 6'd3,
        ST_EVENT_ACCUM   = 6'd4,
        ST_CLS           = 6'd5,
        ST_TOKEN_PREP    = 6'd6,
        ST_TOKEN_WAIT    = 6'd7,
        ST_TOKEN_CAPTURE = 6'd8,
        ST_QKV_INIT      = 6'd9,
        ST_QKV_MAC       = 6'd10,
        ST_ATTN_CLEAR    = 6'd11,
        ST_DOT_INIT      = 6'd12,
        ST_DOT_MAC       = 6'd13,
        ST_ATTN_ACC      = 6'd14,
        ST_FFN_INIT      = 6'd15,
        ST_FFN_MAC       = 6'd16,
        ST_HEAD_INIT     = 6'd17,
        ST_HEAD_MAC      = 6'd18,
        ST_OUT_PREP      = 6'd19,
        ST_OUT_WAIT      = 6'd20,
        ST_OUT_CAPTURE   = 6'd21,
        ST_SCORE_INIT    = 6'd22,
        ST_SCORE_SCAN    = 6'd23,
        ST_RESULT        = 6'd24
    } state_t;

    state_t state;

    (* ram_style = "distributed" *) logic signed [ACC_W-1:0] token_mem [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic signed [ACC_W-1:0] q_mem [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic signed [ACC_W-1:0] k_mem [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic signed [ACC_W-1:0] v_mem [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic signed [15:0] token_v_state [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic signed [15:0] token_u_state [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    (* ram_style = "distributed" *) logic [3:0] token_spike_count [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];

    logic signed [ACC_W-1:0] ctx_vec [0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] ffn_vec [0:EMBED_DIM-1];
    logic signed [15:0] token_v_in;
    logic signed [15:0] token_u_in;
    logic signed [15:0] token_i_in;
    logic signed [15:0] token_v_out;
    logic signed [15:0] token_u_out;
    logic token_spike;

    logic signed [15:0] class_current [0:NUM_CLASSES-1];
    logic signed [15:0] class_v_state [0:NUM_CLASSES-1];
    logic signed [15:0] class_u_state [0:NUM_CLASSES-1];
    logic signed [15:0] class_v_in;
    logic signed [15:0] class_u_in;
    logic signed [15:0] class_i_in;
    logic signed [15:0] class_v_out;
    logic signed [15:0] class_u_out;
    logic class_spike;
    logic [7:0] class_spike_count [0:NUM_CLASSES-1];

    logic [TOKEN_W-1:0] token_idx;
    logic [TOKEN_W-1:0] key_idx;
    logic [DIM_W-1:0] out_dim;
    logic [DIM_W-1:0] in_dim;
    logic [DIM_W-1:0] update_dim;
    logic [CLASS_W-1:0] class_idx;
    logic [STEP_W-1:0] step_idx;
    logic [OUT_STEP_W-1:0] out_step_idx;
    logic [WAIT_W-1:0] wait_count;
    logic [CLEAR_W-1:0] clear_idx;

    logic [PATCH_W-1:0] patch_idx_w;
    logic [PATCH_W-1:0] event_patch;
    logic [7:0] local_x_w;
    logic [7:0] local_y_w;
    logic [7:0] event_local_x;
    logic [7:0] event_local_y;
    logic [1:0] event_channel_hold;
    logic [7:0] event_value_hold;

    logic signed [ACC_W-1:0] q_acc;
    logic signed [ACC_W-1:0] k_acc;
    logic signed [ACC_W-1:0] v_acc;
    logic signed [ACC_W-1:0] dot_acc;
    logic signed [ACC_W-1:0] attn_weight;
    logic signed [ACC_W-1:0] ffn_acc;
    logic signed [ACC_W-1:0] head_acc;
    logic signed [31:0] score_w;

    logic signed [ACC_W-1:0] q_acc_next_w;
    logic signed [ACC_W-1:0] k_acc_next_w;
    logic signed [ACC_W-1:0] v_acc_next_w;
    logic signed [ACC_W-1:0] dot_next_w;
    logic signed [ACC_W-1:0] ffn_next_w;
    logic signed [ACC_W-1:0] head_next_w;
    logic signed [ACC_W-1:0] event_delta_w;
    logic signed [ACC_W-1:0] event_delta_reg;
    logic [3:0] qkv_activity_w;
    logic key_active_w;

    logic [31:0] event_count;
    logic [31:0] active_token_count;
    logic [31:0] skipped_token_count;
    logic [31:0] qkv_mac_count;
    logic [31:0] attention_mac_count;
    logic [31:0] ffn_mac_count;
    logic [31:0] head_mac_count;
    logic [31:0] sqizh_step_count;

    assign event_ready = (state == ST_STREAM);
    assign busy = (state != ST_IDLE);
    assign debug_state = state;
    assign debug_token = {{(8-TOKEN_W){1'b0}}, token_idx};
    assign debug_key = {{(8-TOKEN_W){1'b0}}, key_idx};
    assign debug_dim = {{(8-DIM_W){1'b0}}, out_dim};
    assign debug_dot = dot_acc[31:0];
    assign debug_cls0 = {{16{class_current[0][15]}}, class_current[0]};
    assign debug_cls1 = {{16{class_current[1][15]}}, class_current[1]};
    assign debug_event_count = event_count;
    assign debug_active_tokens = active_token_count;
    assign debug_skipped_tokens = skipped_token_count;
    assign debug_qkv_macs = qkv_mac_count;
    assign debug_attention_macs = attention_mac_count;
    assign debug_ffn_macs = ffn_mac_count;
    assign debug_head_macs = head_mac_count;
    assign debug_sqizh_steps = sqizh_step_count;

    always_comb begin
        patch_idx_w = ((event_y / PATCH_SIZE) * PATCH_COLS) + (event_x / PATCH_SIZE);
        local_x_w = event_x % PATCH_SIZE;
        local_y_w = event_y % PATCH_SIZE;
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

    function automatic logic signed [ACC_W-1:0] relu_acc(input logic signed [ACC_W-1:0] value);
        begin
            relu_acc = value[ACC_W-1] ? '0 : value;
        end
    endfunction

    function automatic int signed patch_coeff(input int ch, input int lx, input int ly, input int dim);
        int raw;
        begin
            raw = ((ch + 1) * 13 + (lx + 2) * 7 + (ly + 3) * 5 + (dim + 1) * 11) % 9;
            patch_coeff = raw - 4;
        end
    endfunction

    function automatic int signed proj_coeff(input int in_i, input int out_i, input int salt);
        int raw;
        begin
            raw = ((in_i + 1) * (salt + 5) + (out_i + 3) * (salt + 7) + salt * 17) % 7;
            proj_coeff = raw - 3;
        end
    endfunction

    function automatic int signed ffn_coeff(input int in_i, input int out_i);
        int raw;
        begin
            raw = ((in_i + 2) * 9 + (out_i + 1) * 5 + 3) % 7;
            ffn_coeff = raw - 3;
        end
    endfunction

    function automatic int signed head_coeff(input int in_i, input int class_i);
        int raw;
        begin
            raw = ((in_i + 1) * 11 + (class_i + 2) * 13 + 19) % 11;
            head_coeff = raw - 5;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] cls_seed(input int dim);
        begin
            cls_seed = ACC_W'(25000 + dim * 256);
        end
    endfunction

    function automatic logic signed [31:0] class_score(
        input logic signed [15:0] current_q8,
        input logic [7:0] spikes
    );
        begin
            class_score = {{16{current_q8[15]}}, current_q8} + $signed({20'd0, spikes, 4'd0});
        end
    endfunction

    always_comb begin
        qkv_activity_w = (token_idx == PATCH_TOKENS[TOKEN_W-1:0]) ? 4'd1 : token_spike_count[token_idx][in_dim];
        q_acc_next_w = q_acc + ($signed({1'b0, qkv_activity_w}) * (proj_coeff(in_dim, out_dim, 1) + 4));
        k_acc_next_w = k_acc + ($signed({1'b0, qkv_activity_w}) * (proj_coeff(in_dim, out_dim, 2) + 4));
        v_acc_next_w = v_acc + ($signed({1'b0, qkv_activity_w}) * (proj_coeff(in_dim, out_dim, 3) + 4));
        dot_next_w = dot_acc + ((q_mem[PATCH_TOKENS][in_dim] * k_mem[key_idx][in_dim]) >>> DOT_SHIFT);
        ffn_next_w = ffn_acc + ((ctx_vec[in_dim] * ffn_coeff(in_dim, out_dim)) >>> FFN_SHIFT);
        head_next_w = head_acc + ((ffn_vec[in_dim] * head_coeff(in_dim, class_idx)) >>> HEAD_SHIFT);
        event_delta_w = ($signed({1'b0, event_value_hold}) * patch_coeff(event_channel_hold, event_local_x, event_local_y, update_dim)) >>> 1;
        score_w = class_score(class_current[class_idx], class_spike_count[class_idx]);
        key_active_w = (key_idx == PATCH_TOKENS[TOKEN_W-1:0]);
        for (int d = 0; d < EMBED_DIM; d++) begin
            key_active_w = key_active_w | (token_spike_count[key_idx][d] != 4'd0);
        end
    end

    sqizh_neuron u_token_sqizh (
        .clk(clk),
        .v_in(token_v_in),
        .u_in(token_u_in),
        .i_in(token_i_in),
        .v_out(token_v_out),
        .u_out(token_u_out),
        .spike(token_spike)
    );

    sqizh_neuron u_class_sqizh (
        .clk(clk),
        .v_in(class_v_in),
        .u_in(class_u_in),
        .i_in(class_i_in),
        .v_out(class_v_out),
        .u_out(class_u_out),
        .spike(class_spike)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            result_valid <= 1'b0;
            predicted_label <= 4'd0;
            predicted_score <= '0;
            token_idx <= '0;
            key_idx <= '0;
            out_dim <= '0;
            in_dim <= '0;
            update_dim <= '0;
            class_idx <= '0;
            step_idx <= '0;
            out_step_idx <= '0;
            wait_count <= '0;
            clear_idx <= '0;
            event_patch <= '0;
            event_local_x <= '0;
            event_local_y <= '0;
            event_channel_hold <= '0;
            event_value_hold <= '0;
            event_delta_reg <= '0;
            q_acc <= '0;
            k_acc <= '0;
            v_acc <= '0;
            dot_acc <= '0;
            attn_weight <= '0;
            ffn_acc <= '0;
            head_acc <= '0;
            event_count <= 32'd0;
            active_token_count <= 32'd0;
            skipped_token_count <= 32'd0;
            qkv_mac_count <= 32'd0;
            attention_mac_count <= 32'd0;
            ffn_mac_count <= 32'd0;
            head_mac_count <= 32'd0;
            sqizh_step_count <= 32'd0;
            token_v_in <= V_RESET_Q8;
            token_u_in <= U_RESET_Q8;
            token_i_in <= 16'sd0;
            class_v_in <= V_RESET_Q8;
            class_u_in <= U_RESET_Q8;
            class_i_in <= 16'sd0;
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
                    token_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    q_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    k_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    v_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    token_v_state[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= V_RESET_Q8;
                    token_u_state[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= U_RESET_Q8;
                    token_spike_count[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= 4'd0;
                    if (clear_idx < EMBED_DIM) begin
                        ctx_vec[clear_idx[DIM_W-1:0]] <= '0;
                        ffn_vec[clear_idx[DIM_W-1:0]] <= '0;
                    end
                    if (clear_idx < NUM_CLASSES) begin
                        class_current[clear_idx[CLASS_W-1:0]] <= 16'sd0;
                        class_v_state[clear_idx[CLASS_W-1:0]] <= V_RESET_Q8;
                        class_u_state[clear_idx[CLASS_W-1:0]] <= U_RESET_Q8;
                        class_spike_count[clear_idx[CLASS_W-1:0]] <= 8'd0;
                    end
                    if (clear_idx == CLEAR_COUNT - 1) begin
                        event_count <= 32'd0;
                        active_token_count <= 32'd0;
                        skipped_token_count <= 32'd0;
                        qkv_mac_count <= 32'd0;
                        attention_mac_count <= 32'd0;
                        ffn_mac_count <= 32'd0;
                        head_mac_count <= 32'd0;
                        sqizh_step_count <= 32'd0;
                        state <= ST_STREAM;
                    end else begin
                        clear_idx <= clear_idx + 1'b1;
                    end
                end

                ST_STREAM: begin
                    if (event_valid && event_ready && event_x < IMAGE_W && event_y < IMAGE_H && event_channel < CHANNELS) begin
                        event_count <= event_count + 1'b1;
                        event_patch <= patch_idx_w;
                        event_local_x <= local_x_w;
                        event_local_y <= local_y_w;
                        event_channel_hold <= event_channel;
                        event_value_hold <= event_value;
                        update_dim <= '0;
                        state <= ST_EVENT_CALC;
                    end else if (frame_done) begin
                        token_idx <= PATCH_TOKENS[TOKEN_W-1:0];
                        out_dim <= '0;
                        state <= ST_CLS;
                    end
                end

                ST_EVENT_CALC: begin
                    event_delta_reg <= event_delta_w;
                    state <= ST_EVENT_ACCUM;
                end

                ST_EVENT_ACCUM: begin
                    token_mem[event_patch][update_dim] <= token_mem[event_patch][update_dim] + event_delta_reg;
                    if (update_dim == EMBED_DIM - 1) begin
                        state <= ST_STREAM;
                    end else begin
                        update_dim <= update_dim + 1'b1;
                        state <= ST_EVENT_CALC;
                    end
                end

                ST_CLS: begin
                    token_mem[token_idx][out_dim] <= cls_seed(out_dim);
                    if (out_dim == EMBED_DIM - 1) begin
                        token_idx <= '0;
                        out_dim <= '0;
                        step_idx <= '0;
                        state <= ST_TOKEN_PREP;
                    end else begin
                        out_dim <= out_dim + 1'b1;
                    end
                end

                ST_TOKEN_PREP: begin
                    token_v_in <= token_v_state[token_idx][out_dim];
                    token_u_in <= token_u_state[token_idx][out_dim];
                    token_i_in <= clamp_i16({{(ACC_W-16){TOKEN_BASE_Q8[15]}}, TOKEN_BASE_Q8} + (token_mem[token_idx][out_dim] >>> TOKEN_INPUT_SHIFT));
                    wait_count <= '0;
                    state <= ST_TOKEN_WAIT;
                end

                ST_TOKEN_WAIT: begin
                    if (wait_count == SQIZH_LATENCY - 1) begin
                        state <= ST_TOKEN_CAPTURE;
                    end else begin
                        wait_count <= wait_count + 1'b1;
                    end
                end

                ST_TOKEN_CAPTURE: begin
                    sqizh_step_count <= sqizh_step_count + 1'b1;
                    token_v_state[token_idx][out_dim] <= token_v_out;
                    token_u_state[token_idx][out_dim] <= token_u_out;
                    token_spike_count[token_idx][out_dim] <= token_spike_count[token_idx][out_dim] + {3'd0, token_spike};
                    if (token_spike) begin
                        active_token_count <= active_token_count + 1'b1;
                    end else begin
                        skipped_token_count <= skipped_token_count + 1'b1;
                    end
                    if (out_dim == EMBED_DIM - 1) begin
                        out_dim <= '0;
                        if (token_idx == TOTAL_TOKENS - 1) begin
                            if (step_idx == NUM_STEPS - 1) begin
                                token_idx <= '0;
                                in_dim <= '0;
                                q_acc <= '0;
                                k_acc <= '0;
                                v_acc <= '0;
                                state <= ST_QKV_INIT;
                            end else begin
                                step_idx <= step_idx + 1'b1;
                                token_idx <= '0;
                                state <= ST_TOKEN_PREP;
                            end
                        end else begin
                            token_idx <= token_idx + 1'b1;
                            state <= ST_TOKEN_PREP;
                        end
                    end else begin
                        out_dim <= out_dim + 1'b1;
                        state <= ST_TOKEN_PREP;
                    end
                end

                ST_QKV_INIT: begin
                    q_acc <= '0;
                    k_acc <= '0;
                    v_acc <= '0;
                    in_dim <= '0;
                    state <= ST_QKV_MAC;
                end

                ST_QKV_MAC: begin
                    if (qkv_activity_w != 4'd0) begin
                        qkv_mac_count <= qkv_mac_count + 3'd3;
                    end
                    if (in_dim == EMBED_DIM - 1) begin
                        q_mem[token_idx][out_dim] <= q_acc_next_w;
                        k_mem[token_idx][out_dim] <= k_acc_next_w;
                        v_mem[token_idx][out_dim] <= v_acc_next_w;
                        q_acc <= '0;
                        k_acc <= '0;
                        v_acc <= '0;
                        in_dim <= '0;
                        if (out_dim == EMBED_DIM - 1) begin
                            out_dim <= '0;
                            if (token_idx == TOTAL_TOKENS - 1) begin
                                state <= ST_ATTN_CLEAR;
                            end else begin
                                token_idx <= token_idx + 1'b1;
                                state <= ST_QKV_INIT;
                            end
                        end else begin
                            out_dim <= out_dim + 1'b1;
                            state <= ST_QKV_INIT;
                        end
                    end else begin
                        q_acc <= q_acc_next_w;
                        k_acc <= k_acc_next_w;
                        v_acc <= v_acc_next_w;
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_ATTN_CLEAR: begin
                    ctx_vec[out_dim] <= '0;
                    if (out_dim == EMBED_DIM - 1) begin
                        key_idx <= '0;
                        in_dim <= '0;
                        dot_acc <= '0;
                        state <= ST_DOT_INIT;
                    end else begin
                        out_dim <= out_dim + 1'b1;
                    end
                end

                ST_DOT_INIT: begin
                    dot_acc <= '0;
                    in_dim <= '0;
                    state <= ST_DOT_MAC;
                end

                ST_DOT_MAC: begin
                    if (key_active_w) begin
                        attention_mac_count <= attention_mac_count + 1'b1;
                    end
                    if (in_dim == EMBED_DIM - 1) begin
                        dot_acc <= dot_next_w;
                        attn_weight <= dot_next_w[ACC_W-1] ? ((-dot_next_w) >>> ATTN_SHIFT) : (dot_next_w >>> ATTN_SHIFT);
                        out_dim <= '0;
                        if (key_active_w && dot_next_w != 0) begin
                            state <= ST_ATTN_ACC;
                        end else if (key_idx == TOTAL_TOKENS - 1) begin
                            out_dim <= '0;
                            in_dim <= '0;
                            state <= ST_FFN_INIT;
                        end else begin
                            key_idx <= key_idx + 1'b1;
                            state <= ST_DOT_INIT;
                        end
                    end else begin
                        dot_acc <= dot_next_w;
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_ATTN_ACC: begin
                    attention_mac_count <= attention_mac_count + 1'b1;
                    ctx_vec[out_dim] <= ctx_vec[out_dim] + ((attn_weight * v_mem[key_idx][out_dim]) >>> ATTN_SHIFT);
                    if (out_dim == EMBED_DIM - 1) begin
                        if (key_idx == TOTAL_TOKENS - 1) begin
                            out_dim <= '0;
                            in_dim <= '0;
                            state <= ST_FFN_INIT;
                        end else begin
                            key_idx <= key_idx + 1'b1;
                            state <= ST_DOT_INIT;
                        end
                    end else begin
                        out_dim <= out_dim + 1'b1;
                    end
                end

                ST_FFN_INIT: begin
                    ffn_acc <= ctx_vec[out_dim];
                    in_dim <= '0;
                    state <= ST_FFN_MAC;
                end

                ST_FFN_MAC: begin
                    ffn_mac_count <= ffn_mac_count + 1'b1;
                    if (in_dim == EMBED_DIM - 1) begin
                        ffn_vec[out_dim] <= relu_acc(ffn_next_w);
                        in_dim <= '0;
                        if (out_dim == EMBED_DIM - 1) begin
                            class_idx <= '0;
                            state <= ST_HEAD_INIT;
                        end else begin
                            out_dim <= out_dim + 1'b1;
                            state <= ST_FFN_INIT;
                        end
                    end else begin
                        ffn_acc <= ffn_next_w;
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_HEAD_INIT: begin
                    head_acc <= '0;
                    in_dim <= '0;
                    state <= ST_HEAD_MAC;
                end

                ST_HEAD_MAC: begin
                    head_mac_count <= head_mac_count + 1'b1;
                    if (in_dim == EMBED_DIM - 1) begin
                        class_current[class_idx] <= clamp_i16({{(ACC_W-16){CLASS_BASE_Q8[15]}}, CLASS_BASE_Q8} + (head_next_w >>> CLASS_CURRENT_SHIFT));
                        head_acc <= '0;
                        in_dim <= '0;
                        if (class_idx == NUM_CLASSES - 1) begin
                            class_idx <= '0;
                            out_step_idx <= '0;
                            state <= ST_OUT_PREP;
                        end else begin
                            class_idx <= class_idx + 1'b1;
                            state <= ST_HEAD_INIT;
                        end
                    end else begin
                        head_acc <= head_next_w;
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_OUT_PREP: begin
                    class_v_in <= class_v_state[class_idx];
                    class_u_in <= class_u_state[class_idx];
                    class_i_in <= class_current[class_idx];
                    wait_count <= '0;
                    state <= ST_OUT_WAIT;
                end

                ST_OUT_WAIT: begin
                    if (wait_count == SQIZH_LATENCY - 1) begin
                        state <= ST_OUT_CAPTURE;
                    end else begin
                        wait_count <= wait_count + 1'b1;
                    end
                end

                ST_OUT_CAPTURE: begin
                    sqizh_step_count <= sqizh_step_count + 1'b1;
                    class_v_state[class_idx] <= class_v_out;
                    class_u_state[class_idx] <= class_u_out;
                    class_spike_count[class_idx] <= class_spike_count[class_idx] + {7'd0, class_spike};
                    if (class_idx == NUM_CLASSES - 1) begin
                        class_idx <= '0;
                        if (out_step_idx == OUT_STEPS - 1) begin
                            state <= ST_SCORE_INIT;
                        end else begin
                            out_step_idx <= out_step_idx + 1'b1;
                            state <= ST_OUT_PREP;
                        end
                    end else begin
                        class_idx <= class_idx + 1'b1;
                        state <= ST_OUT_PREP;
                    end
                end

                ST_SCORE_INIT: begin
                    predicted_label <= 4'd0;
                    predicted_score <= class_score(class_current[0], class_spike_count[0]);
                    class_idx <= 1;
                    state <= ST_SCORE_SCAN;
                end

                ST_SCORE_SCAN: begin
                    if (score_w > predicted_score) begin
                        predicted_score <= score_w;
                        predicted_label <= class_idx[3:0];
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

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end
endmodule

`default_nettype wire
