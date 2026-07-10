`timescale 1ns / 1ps

// Inspectable CIFAR-10 transformer accelerator core.
//
// This is a sequential fixed-point RTL accelerator for the transformer dataflow:
// sparse RGB events -> patch tokens -> Q/K/V -> attention -> feed-forward -> class head.
// It is intentionally small by default so Vivado/XSim waveforms are usable. The
// parameters expose the scaling points for wider embeddings and additional layers.

module cifar10_transformer_accelerator #(
    parameter int IMAGE_W = 32,
    parameter int IMAGE_H = 32,
    parameter int CHANNELS = 3,
    parameter int PATCH_SIZE = 4,
    parameter int PATCH_COLS = IMAGE_W / PATCH_SIZE,
    parameter int PATCH_ROWS = IMAGE_H / PATCH_SIZE,
    parameter int PATCH_TOKENS = PATCH_COLS * PATCH_ROWS,
    parameter int TOTAL_TOKENS = PATCH_TOKENS + 1,
    parameter int EMBED_DIM = 8,
    parameter int NUM_CLASSES = 10,
    parameter int NUM_LAYERS = 1,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    input  logic event_valid,
    output logic event_ready,
    input  logic [4:0] event_x,
    input  logic [4:0] event_y,
    input  logic [1:0] event_channel,
    input  logic [7:0] event_value,
    input  logic frame_done,

    output logic busy,
    output logic result_valid,
    output logic [3:0] predicted_label,
    output logic signed [ACC_W-1:0] predicted_score,

    output logic [5:0] debug_state,
    output logic [7:0] debug_token,
    output logic [7:0] debug_key,
    output logic [7:0] debug_dim,
    output logic signed [ACC_W-1:0] debug_dot,
    output logic signed [ACC_W-1:0] debug_cls0,
    output logic signed [ACC_W-1:0] debug_cls1
);
    typedef enum logic [5:0] {
        ST_IDLE       = 6'd0,
        ST_CLEAR      = 6'd1,
        ST_STREAM     = 6'd2,
        ST_CLS        = 6'd3,
        ST_QKV_INIT   = 6'd4,
        ST_QKV_MAC    = 6'd5,
        ST_ATTN_INIT  = 6'd6,
        ST_DOT_INIT   = 6'd7,
        ST_DOT_MAC    = 6'd8,
        ST_ATTN_ACC   = 6'd9,
        ST_ATTN_NORM  = 6'd10,
        ST_FFN_INIT   = 6'd11,
        ST_FFN_MAC    = 6'd12,
        ST_NEXT_LAYER = 6'd13,
        ST_HEAD_INIT  = 6'd14,
        ST_HEAD_MAC   = 6'd15,
        ST_RESULT     = 6'd16
    } state_t;

    state_t state;

    logic signed [ACC_W-1:0] token_mem [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] q_mem     [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] k_mem     [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] v_mem     [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] ctx_mem   [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] ffn_mem   [0:TOTAL_TOKENS-1][0:EMBED_DIM-1];
    logic signed [ACC_W-1:0] class_score [0:NUM_CLASSES-1];

    logic [$clog2(TOTAL_TOKENS)-1:0] token_idx;
    logic [$clog2(TOTAL_TOKENS)-1:0] key_idx;
    logic [$clog2(EMBED_DIM)-1:0] out_dim;
    logic [$clog2(EMBED_DIM)-1:0] in_dim;
    logic [$clog2(NUM_CLASSES)-1:0] class_idx;
    logic [$clog2(NUM_LAYERS+1)-1:0] layer_idx;
    logic signed [ACC_W-1:0] dot_acc;
    logic signed [ACC_W-1:0] attn_weight;
    logic signed [ACC_W-1:0] denom_acc;
    logic signed [ACC_W-1:0] head_acc;
    logic [$clog2(TOTAL_TOKENS*EMBED_DIM+1)-1:0] clear_idx;

    logic [$clog2(PATCH_TOKENS)-1:0] patch_idx_w;
    logic [2:0] local_x_w;
    logic [2:0] local_y_w;

    assign local_x_w = event_x[2:0];
    assign local_y_w = event_y[2:0];
    assign patch_idx_w = (($clog2(PATCH_TOKENS))'(event_y / PATCH_SIZE) * PATCH_COLS)
                       + ($clog2(PATCH_TOKENS))'(event_x / PATCH_SIZE);

    assign event_ready = (state == ST_STREAM);
    assign busy = (state != ST_IDLE);
    assign debug_state = state;
    assign debug_token = 8'(token_idx);
    assign debug_key = 8'(key_idx);
    assign debug_dim = 8'(out_dim);
    assign debug_dot = dot_acc;
    assign debug_cls0 = class_score[0];
    assign debug_cls1 = class_score[1];

    function automatic int signed patch_coeff(
        input int ch,
        input int lx,
        input int ly,
        input int dim
    );
        int raw;
        begin
            raw = ((ch + 1) * 17 + (lx + 3) * 11 + (ly + 5) * 7 + (dim + 1) * 13) % 17;
            patch_coeff = raw - 8;
        end
    endfunction

    function automatic int signed proj_coeff(
        input int in_i,
        input int out_i,
        input int salt
    );
        int raw;
        begin
            raw = ((in_i + 1) * (salt + 3) + (out_i + 2) * (salt + 5) + salt * 19) % 15;
            proj_coeff = raw - 7;
        end
    endfunction

    function automatic logic signed [ACC_W-1:0] relu_acc(input logic signed [ACC_W-1:0] value);
        begin
            relu_acc = value[ACC_W-1] ? '0 : value;
        end
    endfunction

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
            class_idx <= '0;
            layer_idx <= '0;
            dot_acc <= '0;
            attn_weight <= '0;
            denom_acc <= '0;
            head_acc <= '0;
            clear_idx <= '0;
            for (int t = 0; t < TOTAL_TOKENS; t++) begin
                for (int e = 0; e < EMBED_DIM; e++) begin
                    token_mem[t][e] <= '0;
                    q_mem[t][e] <= '0;
                    k_mem[t][e] <= '0;
                    v_mem[t][e] <= '0;
                    ctx_mem[t][e] <= '0;
                    ffn_mem[t][e] <= '0;
                end
            end
            for (int c = 0; c < NUM_CLASSES; c++) begin
                class_score[c] <= '0;
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
                    token_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    q_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    k_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    v_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    ctx_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    ffn_mem[clear_idx / EMBED_DIM][clear_idx % EMBED_DIM] <= '0;
                    if (clear_idx == TOTAL_TOKENS * EMBED_DIM - 1) begin
                        state <= ST_STREAM;
                    end else begin
                        clear_idx <= clear_idx + 1'b1;
                    end
                end

                ST_STREAM: begin
                    if (event_valid && event_ready && event_x < IMAGE_W && event_y < IMAGE_H && event_channel < CHANNELS) begin
                        for (int e = 0; e < EMBED_DIM; e++) begin
                            token_mem[patch_idx_w][e] <= token_mem[patch_idx_w][e]
                                + ((ACC_W'($signed({1'b0, event_value})) * patch_coeff(event_channel, local_x_w, local_y_w, e)) >>> 3);
                        end
                    end
                    if (frame_done) begin
                        token_idx <= PATCH_TOKENS[$clog2(TOTAL_TOKENS)-1:0];
                        out_dim <= '0;
                        state <= ST_CLS;
                    end
                end

                ST_CLS: begin
                    token_mem[token_idx][out_dim] <= ACC_W'(32 + int'(out_dim) * 3);
                    if (out_dim == EMBED_DIM - 1) begin
                        token_idx <= '0;
                        out_dim <= '0;
                        in_dim <= '0;
                        layer_idx <= '0;
                        state <= ST_QKV_INIT;
                    end else begin
                        out_dim <= out_dim + 1'b1;
                    end
                end

                ST_QKV_INIT: begin
                    q_mem[token_idx][out_dim] <= '0;
                    k_mem[token_idx][out_dim] <= '0;
                    v_mem[token_idx][out_dim] <= '0;
                    in_dim <= '0;
                    state <= ST_QKV_MAC;
                end

                ST_QKV_MAC: begin
                    q_mem[token_idx][out_dim] <= q_mem[token_idx][out_dim]
                        + ((token_mem[token_idx][in_dim] * proj_coeff(in_dim, out_dim, 1)) >>> 5);
                    k_mem[token_idx][out_dim] <= k_mem[token_idx][out_dim]
                        + ((token_mem[token_idx][in_dim] * proj_coeff(in_dim, out_dim, 5)) >>> 5);
                    v_mem[token_idx][out_dim] <= v_mem[token_idx][out_dim]
                        + ((token_mem[token_idx][in_dim] * proj_coeff(in_dim, out_dim, 9)) >>> 5);

                    if (in_dim == EMBED_DIM - 1) begin
                        if (out_dim == EMBED_DIM - 1) begin
                            if (token_idx == TOTAL_TOKENS - 1) begin
                                token_idx <= '0;
                                key_idx <= '0;
                                out_dim <= '0;
                                state <= ST_ATTN_INIT;
                            end else begin
                                token_idx <= token_idx + 1'b1;
                                out_dim <= '0;
                                state <= ST_QKV_INIT;
                            end
                        end else begin
                            out_dim <= out_dim + 1'b1;
                            state <= ST_QKV_INIT;
                        end
                    end else begin
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_ATTN_INIT: begin
                    for (int e = 0; e < EMBED_DIM; e++) begin
                        ctx_mem[token_idx][e] <= '0;
                    end
                    denom_acc <= '0;
                    key_idx <= '0;
                    state <= ST_DOT_INIT;
                end

                ST_DOT_INIT: begin
                    dot_acc <= '0;
                    in_dim <= '0;
                    state <= ST_DOT_MAC;
                end

                ST_DOT_MAC: begin
                    dot_acc <= dot_acc + ((q_mem[token_idx][in_dim] * k_mem[key_idx][in_dim]) >>> 6);
                    if (in_dim == EMBED_DIM - 1) begin
                        state <= ST_ATTN_ACC;
                    end else begin
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_ATTN_ACC: begin
                    attn_weight <= relu_acc(dot_acc >>> 3);
                    denom_acc <= denom_acc + relu_acc(dot_acc >>> 3) + 1;
                    for (int e = 0; e < EMBED_DIM; e++) begin
                        ctx_mem[token_idx][e] <= ctx_mem[token_idx][e]
                            + (((relu_acc(dot_acc >>> 3) + 1) * v_mem[key_idx][e]) >>> 8);
                    end
                    if (key_idx == TOTAL_TOKENS - 1) begin
                        out_dim <= '0;
                        state <= ST_ATTN_NORM;
                    end else begin
                        key_idx <= key_idx + 1'b1;
                        state <= ST_DOT_INIT;
                    end
                end

                ST_ATTN_NORM: begin
                    if (denom_acc > 0) begin
                        ctx_mem[token_idx][out_dim] <= ctx_mem[token_idx][out_dim] + token_mem[token_idx][out_dim];
                    end else begin
                        ctx_mem[token_idx][out_dim] <= v_mem[token_idx][out_dim] + token_mem[token_idx][out_dim];
                    end
                    if (out_dim == EMBED_DIM - 1) begin
                        if (token_idx == TOTAL_TOKENS - 1) begin
                            token_idx <= '0;
                            out_dim <= '0;
                            in_dim <= '0;
                            state <= ST_FFN_INIT;
                        end else begin
                            token_idx <= token_idx + 1'b1;
                            key_idx <= '0;
                            state <= ST_ATTN_INIT;
                        end
                    end else begin
                        out_dim <= out_dim + 1'b1;
                    end
                end

                ST_FFN_INIT: begin
                    ffn_mem[token_idx][out_dim] <= '0;
                    in_dim <= '0;
                    state <= ST_FFN_MAC;
                end

                ST_FFN_MAC: begin
                    ffn_mem[token_idx][out_dim] <= ffn_mem[token_idx][out_dim]
                        + ((relu_acc(ctx_mem[token_idx][in_dim]) * proj_coeff(in_dim, out_dim, 13)) >>> 6);
                    if (in_dim == EMBED_DIM - 1) begin
                        token_mem[token_idx][out_dim] <= ctx_mem[token_idx][out_dim] + ffn_mem[token_idx][out_dim];
                        if (out_dim == EMBED_DIM - 1) begin
                            if (token_idx == TOTAL_TOKENS - 1) begin
                                state <= ST_NEXT_LAYER;
                            end else begin
                                token_idx <= token_idx + 1'b1;
                                out_dim <= '0;
                                state <= ST_FFN_INIT;
                            end
                        end else begin
                            out_dim <= out_dim + 1'b1;
                            state <= ST_FFN_INIT;
                        end
                    end else begin
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_NEXT_LAYER: begin
                    if (layer_idx == NUM_LAYERS - 1) begin
                        class_idx <= '0;
                        out_dim <= '0;
                        state <= ST_HEAD_INIT;
                    end else begin
                        layer_idx <= layer_idx + 1'b1;
                        token_idx <= '0;
                        out_dim <= '0;
                        in_dim <= '0;
                        state <= ST_QKV_INIT;
                    end
                end

                ST_HEAD_INIT: begin
                    class_score[class_idx] <= '0;
                    in_dim <= '0;
                    state <= ST_HEAD_MAC;
                end

                ST_HEAD_MAC: begin
                    class_score[class_idx] <= class_score[class_idx]
                        + ((token_mem[TOTAL_TOKENS-1][in_dim] * proj_coeff(in_dim, class_idx, 23)) >>> 5);
                    if (in_dim == EMBED_DIM - 1) begin
                        if (class_idx == NUM_CLASSES - 1) begin
                            state <= ST_RESULT;
                        end else begin
                            class_idx <= class_idx + 1'b1;
                            state <= ST_HEAD_INIT;
                        end
                    end else begin
                        in_dim <= in_dim + 1'b1;
                    end
                end

                ST_RESULT: begin
                    predicted_label <= 4'd0;
                    predicted_score <= class_score[0];
                    for (int c = 1; c < NUM_CLASSES; c++) begin
                        if (class_score[c] >= predicted_score) begin
                            predicted_label <= c[3:0];
                            predicted_score <= class_score[c];
                        end
                    end
                    result_valid <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (IMAGE_W != 32 || IMAGE_H != 32 || CHANNELS != 3) begin
            $error("cifar10_transformer_accelerator expects CIFAR-10 RGB 32x32 input.");
        end
        if (PATCH_SIZE != 4) begin
            $warning("Default debug waveforms assume patch_size=4.");
        end
    end
endmodule
