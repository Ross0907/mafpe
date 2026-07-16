`timescale 1ns / 1ps
`default_nettype none

module neuraedge_universal_core (
    input  wire clk,
    input  wire rst_n,
    input  wire command_valid,
    output logic command_ready,
    input  wire [127:0] command_data,
    output logic command_done,
    output logic status_busy,
    output logic status_error,
    output logic [3:0] status_active_opcode,
    output logic [31:0] status_command_count,
    output logic [31:0] status_input_beats,
    output logic [31:0] status_output_beats,
    output wire [11:0] debug_tensor_loaded_activations,
    output wire [11:0] debug_tensor_loaded_weights,
    output wire [11:0] debug_tensor_active_issues,
    output wire [31:0] debug_tensor_compute_cycles,
    output wire [31:0] debug_tensor_skipped_issues,
    output wire [63:0] debug_tensor_dense_macs,
    output wire [63:0] debug_tensor_spike_adds,
    output wire [31:0] debug_layernorm_cycles,
    output wire signed [31:0] debug_layernorm_mean,
    output wire [31:0] debug_layernorm_variance,
    output wire [31:0] debug_layernorm_sigma,
    output wire [31:0] debug_layernorm_reciprocal,
    output wire [31:0] debug_softmax_cycles,
    output wire signed [23:0] debug_softmax_maximum,
    output wire [23:0] debug_softmax_denominator,
    output wire [31:0] debug_softmax_reciprocal,
    output wire [63:0] debug_lif_total_spikes,
    output wire [31:0] debug_optimizer_updates,
    output wire [8:0] debug_lut_loaded_entries,
    output wire [31:0] debug_training_tiles,
    output wire [31:0] debug_training_cycles,

    input  wire s_axis_tvalid,
    output logic s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire s_axis_tlast,
    output logic m_axis_tvalid,
    input  wire m_axis_tready,
    output logic [63:0] m_axis_tdata,
    output logic m_axis_tlast
);
    import neuraedge_command_pkg::*;

    typedef enum logic [5:0] {
        ST_IDLE = 6'd0,
        ST_TENSOR = 6'd1,
        ST_ALPHA_LOAD = 6'd2,
        ST_ALPHA_START = 6'd3,
        ST_ALPHA_OUTPUT = 6'd4,
        ST_VECTOR_LOAD = 6'd5,
        ST_VECTOR_START = 6'd6,
        ST_VECTOR_OUTPUT = 6'd7,
        ST_LAYERNORM_START = 6'd8,
        ST_LAYERNORM_LOAD0 = 6'd9,
        ST_LAYERNORM_LOAD1 = 6'd10,
        ST_LAYERNORM_SEND = 6'd11,
        ST_LAYERNORM_OUTPUT = 6'd12,
        ST_SOFTMAX_START = 6'd13,
        ST_SOFTMAX_LOAD = 6'd14,
        ST_SOFTMAX_OUTPUT = 6'd15,
        ST_LIF_LOAD = 6'd16,
        ST_LIF_CLEAR = 6'd17,
        ST_LIF_START = 6'd18,
        ST_LIF_OUTPUT = 6'd19,
        ST_OPTIMIZER_LOAD = 6'd20,
        ST_OPTIMIZER_START = 6'd21,
        ST_OPTIMIZER_OUTPUT = 6'd22,
        ST_LUT_LOAD = 6'd23,
        ST_LUT_EVAL_LOAD = 6'd24,
        ST_LUT_EVAL_START = 6'd25,
        ST_LUT_EVAL_OUTPUT = 6'd26,
        ST_LUT_CLEAR = 6'd27,
        ST_ALPHA_DRAIN = 6'd28,
        ST_OPTIMIZER_COMMIT = 6'd29,
        ST_VECTOR_DRAIN = 6'd30,
        ST_DECODE_COMMAND = 6'd31,
        ST_TRAIN_START = 6'd32,
        ST_TRAIN_RUN = 6'd33,
        ST_OUTPUT_DRAIN = 6'd34
    } state_t;

    // One-hot recoding turns every engine-select decision into a single state
    // bit instead of a shared five-bit decode with hundreds of loads.
    (* fsm_encoding = "one_hot", extract_enable = "no" *) state_t state;
    logic status_error_r;
    (* dont_touch = "yes" *) logic error_event_r;
    (* dont_touch = "yes" *) logic clear_error_r;
    (* dont_touch = "yes" *) logic stream_protocol_error_event_r;
    (* dont_touch = "yes" *) logic clear_counters_event_r;
    (* dont_touch = "yes" *) logic command_done_pending_r;
    (* dont_touch = "yes" *) logic input_beat_accepted_r;
    (* dont_touch = "yes" *) logic output_beat_accepted_r;
    logic [7:0] stream_beat_index;
    logic [11:0] stream_item_index;
    logic [11:0] stream_items_remaining_r;
    logic stream_last_pending_r;
    logic [3:0] output_word_index;
    wire [9:0] lif_trace_bit_offset =
        ({6'd0, output_word_index} - 10'd1) << 6;
    logic [108:0] captured_command_r;
    logic [3:0] train_reserved_partial_r;
    logic train_reserved_nonzero_r;

    typedef enum logic [3:0] {
        RESULT_TENSOR = 4'd0,
        RESULT_ALPHA = 4'd1,
        RESULT_VECTOR = 4'd2,
        RESULT_LAYERNORM = 4'd3,
        RESULT_SOFTMAX = 4'd4,
        RESULT_LIF = 4'd5,
        RESULT_OPTIMIZER = 4'd6,
        RESULT_TRAIN = 4'd7,
        RESULT_LUT = 4'd8
    } result_source_t;

    // Each engine writes a private result register before the shared output
    // mux.  The following elastic register then arbitrates those local banks.
    // Splitting formatting and arbitration across clocks removes the former
    // child-to-wrapper nine-way combinational path while preserving one result
    // beat per clock after the two stages fill.
    logic result_input_valid;
    logic result_input_last;
    logic result_prestage_valid_r;
    result_source_t result_prestage_source_r;
    logic result_prestage_last_r;
    logic [63:0] result_tensor_data_r;
    logic [63:0] result_alpha_data_r;
    logic [63:0] result_vector_data_r;
    logic [63:0] result_layernorm_data_r;
    logic [63:0] result_softmax_data_r;
    logic [63:0] result_lif_data_r;
    logic [63:0] result_optimizer_data_r;
    logic [63:0] result_train_data_r;
    logic [63:0] result_lut_data_r;
    logic result_output_valid_r;
    logic [63:0] result_output_data_r;
    logic result_output_last_r;
    logic command_had_output_r;
    logic final_output_accepted_r;
    logic completion_error_r;

    wire result_output_ready = !result_output_valid_r || m_axis_tready;
    wire result_prestage_ready = !result_prestage_valid_r
        || result_output_ready;
    wire result_input_accept = result_input_valid && result_prestage_ready;
    wire result_output_accept = result_output_valid_r && m_axis_tready;

    wire command_is_tensor = captured_command_r[3:0] >= CMD_LOAD_ACTIVATIONS
        && captured_command_r[3:0] <= CMD_COMPUTE_TERNARY;
    wire command_routes_tensor = command_is_tensor
        || captured_command_r[3:0] == CMD_CLEAR_STATUS;

    logic tensor_command_valid;
    wire tensor_command_ready;
    wire tensor_command_done;
    wire tensor_status_error;
    logic tensor_s_valid;
    wire tensor_s_ready;
    logic tensor_m_ready;
    wire tensor_m_valid;
    wire [63:0] tensor_m_data;
    wire tensor_m_last;

    neuraedge_buffered_tensor_core u_tensor_core (
        .clk(clk), .rst_n(rst_n),
        .command_valid(tensor_command_valid), .command_ready(tensor_command_ready),
        .command_data(captured_command_r[41:0]), .command_done(tensor_command_done),
        .status_error(tensor_status_error), .status_busy(),
        .s_axis_tvalid(tensor_s_valid), .s_axis_tready(tensor_s_ready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(tensor_m_valid), .m_axis_tready(tensor_m_ready),
        .m_axis_tdata(tensor_m_data), .m_axis_tlast(tensor_m_last),
        .status_loaded_activations(debug_tensor_loaded_activations),
        .status_loaded_weights(debug_tensor_loaded_weights),
        .status_active_issues(debug_tensor_active_issues),
        .status_input_beats(),
        .status_output_beats(),
        .status_compute_cycles(debug_tensor_compute_cycles),
        .status_skipped_issues(debug_tensor_skipped_issues),
        .status_dense_macs(debug_tensor_dense_macs),
        .status_spike_adds(debug_tensor_spike_adds)
    );

    logic [255:0] alpha_query_r;
    logic [255:0] alpha_key_r;
    logic [8:0] alpha_active_bits_r;
    logic [8:0] alpha_q8_r;
    logic alpha_load_valid_r;
    logic [2:0] alpha_load_index_r;
    logic [63:0] alpha_load_data_r;
    logic alpha_load_final_r;
    logic alpha_load_last_r;
    logic alpha_input_valid;
    wire alpha_input_ready;
    wire alpha_output_valid;
    logic alpha_output_ready;
    wire [17:0] alpha_score;
    wire [8:0] alpha_spike_matches;
    wire [8:0] alpha_zero_matches;
    wire alpha_config_error;

    neuraedge_alpha_xnor u_alpha_xnor (
        .clk(clk), .rst_n(rst_n),
        .input_valid(alpha_input_valid), .input_ready(alpha_input_ready),
        .query_spikes(alpha_query_r), .key_spikes(alpha_key_r),
        .active_bits(alpha_active_bits_r), .alpha_q8(alpha_q8_r),
        .output_valid(alpha_output_valid), .output_ready(alpha_output_ready),
        .score_q8(alpha_score), .spike_matches(alpha_spike_matches),
        .zero_matches(alpha_zero_matches), .config_error(alpha_config_error)
    );

    logic [3:0] vector_operation_r;
    logic [255:0] vector_operand_a_r;
    logic [255:0] vector_operand_b_r;
    logic signed [15:0] vector_multiplier_r;
    logic [4:0] vector_shift_r;
    logic signed [15:0] vector_zero_point_r;
    logic signed [31:0] vector_threshold_r;
    logic [31:0] vector_window_r;
    logic signed [31:0] vector_clamp_min_r;
    logic signed [31:0] vector_clamp_max_r;
    (* dont_touch = "yes" *) logic vector_load_valid_r;
    logic [3:0] vector_load_index_r;
    logic [63:0] vector_load_data_r;
    logic vector_load_final_r;
    logic vector_load_last_r;
    logic vector_input_valid;
    wire vector_input_ready;
    wire vector_output_valid;
    logic vector_output_ready;
    wire [255:0] vector_result;
    wire vector_config_error;

    neuraedge_vector_alu u_vector_alu (
        .clk(clk), .rst_n(rst_n),
        .input_valid(vector_input_valid), .input_ready(vector_input_ready),
        .operation(vector_operation_r), .operand_a(vector_operand_a_r),
        .operand_b(vector_operand_b_r), .multiplier_q15(vector_multiplier_r),
        .right_shift(vector_shift_r), .output_zero_point(vector_zero_point_r),
        .threshold(vector_threshold_r), .surrogate_window(vector_window_r),
        .clamp_minimum(vector_clamp_min_r), .clamp_maximum(vector_clamp_max_r),
        .output_valid(vector_output_valid), .output_ready(vector_output_ready),
        .result(vector_result), .config_error(vector_config_error)
    );

    logic [11:0] ln_count_r;
    logic [31:0] ln_epsilon_r;
    logic [63:0] ln_sample_word0_r;
    logic signed [23:0] ln_sample_value_r;
    logic signed [23:0] ln_sample_gamma_r;
    logic signed [23:0] ln_sample_beta_r;
    logic ln_start_valid;
    wire ln_start_ready;
    logic ln_sample_valid;
    wire ln_sample_ready;
    logic ln_sample_last;
    wire ln_result_valid;
    logic ln_result_ready;
    wire signed [31:0] ln_result_value;
    wire [11:0] ln_result_index;
    wire ln_result_last;
    wire ln_config_error;
    logic ln_start_accept_r;

    neuraedge_integer_layernorm u_layernorm (
        .clk(clk), .rst_n(rst_n),
        .start_valid(ln_start_valid), .start_ready(ln_start_ready),
        .feature_count(ln_count_r), .epsilon(ln_epsilon_r),
        .sample_valid(ln_sample_valid), .sample_ready(ln_sample_ready),
        .sample_value(ln_sample_value_r), .sample_gamma_q15(ln_sample_gamma_r),
        .sample_beta(ln_sample_beta_r), .sample_last(ln_sample_last),
        .result_valid(ln_result_valid), .result_ready(ln_result_ready),
        .result_value(ln_result_value), .result_index(ln_result_index),
        .result_last(ln_result_last), .config_error(ln_config_error),
        .cycle_count(debug_layernorm_cycles),
        .debug_mean(debug_layernorm_mean),
        .debug_variance(debug_layernorm_variance),
        .debug_sigma(debug_layernorm_sigma),
        .debug_reciprocal_q31(debug_layernorm_reciprocal)
    );

    logic [8:0] softmax_count_r;
    logic [4:0] softmax_shift_r;
    logic softmax_start_accept_r;
    logic softmax_start_valid;
    wire softmax_start_ready;
    logic softmax_score_valid;
    wire softmax_score_ready;
    logic signed [23:0] softmax_score_value;
    logic softmax_score_last;
    wire softmax_probability_valid;
    logic softmax_probability_ready;
    wire [15:0] softmax_probability;
    wire [8:0] softmax_probability_index;
    wire softmax_probability_last;
    wire softmax_config_error;

    neuraedge_log_softmax u_softmax (
        .clk(clk), .rst_n(rst_n),
        .start_valid(softmax_start_valid), .start_ready(softmax_start_ready),
        .item_count(softmax_count_r), .difference_shift(softmax_shift_r),
        .score_valid(softmax_score_valid), .score_ready(softmax_score_ready),
        .score_value(softmax_score_value), .score_last(softmax_score_last),
        .probability_valid(softmax_probability_valid),
        .probability_ready(softmax_probability_ready),
        .probability_q15(softmax_probability),
        .probability_index(softmax_probability_index),
        .probability_last(softmax_probability_last), .config_error(softmax_config_error),
        .cycle_count(debug_softmax_cycles),
        .debug_maximum_score(debug_softmax_maximum),
        .debug_denominator(debug_softmax_denominator),
        .debug_reciprocal_q31(debug_softmax_reciprocal)
    );

    logic [511:0] lif_current_r;
    logic [4:0] lif_leak_shift_r;
    logic signed [23:0] lif_threshold_r;
    logic signed [23:0] lif_reset_r;
    logic lif_clear_requested_r;
    logic [7:0] lif_load_word_select_r;
    logic lif_load_pending_r;
    logic [2:0] lif_load_index_r;
    logic [63:0] lif_load_data_r;
    logic lif_load_last_r;
    logic lif_load_final_r;
    logic lif_clear_state;
    logic lif_input_valid;
    wire lif_input_ready;
    wire lif_output_valid;
    logic lif_output_ready;
    wire [31:0] lif_spikes;
    wire [32*24-1:0] lif_pre_reset_membrane;
    wire [31:0] lif_timesteps;
    logic lif_trace_requested_r;

    neuraedge_lif_bank u_lif_bank (
        .clk(clk), .rst_n(rst_n), .clear_state(lif_clear_state),
        .clear_counters(clear_counters_event_r),
        .input_valid(lif_input_valid), .input_ready(lif_input_ready),
        .current_vector(lif_current_r), .leak_shift(lif_leak_shift_r),
        .threshold(lif_threshold_r), .reset_value(lif_reset_r),
        .output_valid(lif_output_valid), .output_ready(lif_output_ready),
        .spikes(lif_spikes), .membrane_vector(),
        .pre_reset_membrane_vector(lif_pre_reset_membrane),
        .timestep_count(lif_timesteps),
        .total_spike_count(debug_lif_total_spikes)
    );

    logic [127:0] optimizer_weight_r;
    logic [191:0] optimizer_gradient_r;
    logic [191:0] optimizer_velocity_r;
    logic [17:0] optimizer_lr_r;
    logic [16:0] optimizer_momentum_r;
    logic [16:0] optimizer_decay_r;
    logic [4:0] optimizer_update_shift_r;
    logic [63:0] optimizer_load_data_r;
    logic optimizer_load_last_r;
    logic [2:0] optimizer_load_index_r;
    logic optimizer_load_final_r;
    (* dont_touch = "yes" *) logic optimizer_commit_valid_r;
    logic optimizer_input_valid;
    wire optimizer_input_ready;
    wire optimizer_output_valid;
    logic optimizer_output_ready;
    wire [127:0] optimizer_updated_weight;
    wire [191:0] optimizer_updated_velocity;
    wire optimizer_config_error;

    neuraedge_sgd_optimizer8 u_optimizer (
        .clk(clk), .rst_n(rst_n), .clear_counters(clear_counters_event_r),
        .input_valid(optimizer_input_valid), .input_ready(optimizer_input_ready),
        .weight_vector(optimizer_weight_r), .gradient_vector(optimizer_gradient_r),
        .velocity_vector(optimizer_velocity_r), .learning_rate_q16(optimizer_lr_r),
        .momentum_q16(optimizer_momentum_r), .weight_decay_q16(optimizer_decay_r),
        .update_right_shift(optimizer_update_shift_r),
        .output_valid(optimizer_output_valid), .output_ready(optimizer_output_ready),
        .updated_weight_vector(optimizer_updated_weight),
        .updated_velocity_vector(optimizer_updated_velocity),
        .config_error(optimizer_config_error),
        .update_count(debug_optimizer_updates)
    );

    logic train_start_valid;
    wire train_start_ready;
    logic train_s_valid;
    wire train_s_ready;
    wire train_m_valid;
    logic train_m_ready;
    wire [63:0] train_m_data;
    wire train_m_last;
    wire train_done;
    wire train_config_error;
    wire train_busy;

    neuraedge_linear_backward8 u_linear_backward (
        .clk(clk), .rst_n(rst_n), .clear_counters(clear_counters_event_r),
        .start_valid(train_start_valid), .start_ready(train_start_ready),
        .active_rows(captured_command_r[7:4]),
        .active_inputs(captured_command_r[11:8]),
        .active_outputs(captured_command_r[15:12]),
        .input_gradient_shift(captured_command_r[20:16]),
        .weight_gradient_shift(captured_command_r[25:21]),
        .gradient_mode(captured_command_r[27:26]),
        .accumulate_result(captured_command_r[28]),
        .suppress_output(captured_command_r[29]),
        .reserved_nonzero(train_reserved_nonzero_r),
        .s_axis_tvalid(train_s_valid), .s_axis_tready(train_s_ready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(train_m_valid), .m_axis_tready(train_m_ready),
        .m_axis_tdata(train_m_data), .m_axis_tlast(train_m_last),
        .busy(train_busy), .done(train_done), .config_error(train_config_error),
        .completed_tiles(debug_training_tiles),
        .active_cycles(debug_training_cycles)
    );

    logic lut_clear;
    logic lut_load_valid;
    wire lut_load_ready;
    logic [7:0] lut_load_address;
    logic signed [15:0] lut_load_value;
    logic lut_load_pending_r;
    logic [7:0] lut_load_address_r;
    logic signed [15:0] lut_load_value_r;
    logic lut_load_last_r;
    logic lut_load_final_r;
    logic lut_input_valid;
    wire lut_input_ready;
    logic [63:0] lut_indices_r;
    wire lut_output_valid;
    logic lut_output_ready;
    wire [127:0] lut_output_values;
    wire lut_config_error;
    logic lut_clear_requested_r;
    logic decode_error_pending_r;

    neuraedge_activation_lut8 u_activation_lut (
        .clk(clk), .rst_n(rst_n), .clear_table(lut_clear),
        .load_valid(lut_load_valid), .load_ready(lut_load_ready),
        .load_address(lut_load_address), .load_value(lut_load_value),
        .input_valid(lut_input_valid), .input_ready(lut_input_ready),
        .input_indices(lut_indices_r), .output_valid(lut_output_valid),
        .output_ready(lut_output_ready), .output_values(lut_output_values),
        .config_error(lut_config_error),
        .loaded_entries(debug_lut_loaded_entries)
    );

    // Keep each engine's transfer event local.  A shared OR-of-valids is fine
    // for the single accounting bit, but using it in unrelated FSM branches
    // creates false cross-engine control cones after one-hot optimization.
    wire tensor_result_accept = state == ST_TENSOR
        && tensor_m_valid && result_prestage_ready;
    wire alpha_result_accept = state == ST_ALPHA_OUTPUT
        && alpha_output_valid && result_prestage_ready;
    wire vector_result_accept = state == ST_VECTOR_OUTPUT
        && vector_output_valid && result_prestage_ready;
    wire layernorm_result_accept = state == ST_LAYERNORM_OUTPUT
        && ln_result_valid && result_prestage_ready;
    wire softmax_result_accept = state == ST_SOFTMAX_OUTPUT
        && softmax_probability_valid && result_prestage_ready;
    wire lif_result_accept = state == ST_LIF_OUTPUT
        && lif_output_valid && result_prestage_ready;
    wire optimizer_result_accept = state == ST_OPTIMIZER_OUTPUT
        && optimizer_output_valid && result_prestage_ready;
    wire train_result_accept = state == ST_TRAIN_RUN
        && train_m_valid && result_prestage_ready;
    wire lut_result_accept = state == ST_LUT_EVAL_OUTPUT
        && lut_output_valid && result_prestage_ready;

    always_comb begin
        command_ready = state == ST_IDLE;
        status_busy = state != ST_IDLE || train_busy;
        status_error = status_error_r || tensor_status_error;
        s_axis_tready = 1'b0;
        m_axis_tvalid = result_output_valid_r;
        m_axis_tdata = result_output_data_r;
        m_axis_tlast = result_output_last_r;
        result_input_valid = 1'b0;
        result_input_last = 1'b0;

        tensor_command_valid = state == ST_DECODE_COMMAND && command_routes_tensor;
        tensor_s_valid = 1'b0;
        tensor_m_ready = 1'b0;
        alpha_input_valid = 1'b0;
        alpha_output_ready = 1'b0;
        vector_input_valid = 1'b0;
        vector_output_ready = 1'b0;
        ln_start_valid = 1'b0;
        ln_sample_valid = 1'b0;
        ln_sample_last = stream_last_pending_r;
        ln_result_ready = 1'b0;
        softmax_start_valid = 1'b0;
        softmax_score_valid = 1'b0;
        softmax_score_value = s_axis_tdata[23:0];
        softmax_score_last = stream_last_pending_r;
        softmax_probability_ready = 1'b0;
        lif_clear_state = 1'b0;
        lif_input_valid = 1'b0;
        lif_output_ready = 1'b0;
        optimizer_input_valid = 1'b0;
        optimizer_output_ready = 1'b0;
        train_start_valid = 1'b0;
        train_s_valid = 1'b0;
        train_m_ready = 1'b0;
        lut_clear = 1'b0;
        lut_load_valid = 1'b0;
        lut_load_address = lut_load_address_r;
        lut_load_value = lut_load_value_r;
        lut_input_valid = 1'b0;
        lut_output_ready = 1'b0;

        unique case (state)
            ST_TENSOR: begin
                tensor_s_valid = s_axis_tvalid;
                s_axis_tready = tensor_s_ready;
                result_input_valid = tensor_m_valid;
                result_input_last = tensor_m_last;
                tensor_m_ready = result_prestage_ready;
            end
            ST_ALPHA_LOAD: s_axis_tready = !(alpha_load_valid_r && alpha_load_final_r);
            ST_VECTOR_LOAD: s_axis_tready = !vector_load_final_r;
            ST_OPTIMIZER_LOAD, ST_LUT_EVAL_LOAD: s_axis_tready = 1'b1;
            ST_LIF_LOAD: s_axis_tready = !lif_load_pending_r;
            ST_ALPHA_START: alpha_input_valid = 1'b1;
            ST_ALPHA_OUTPUT: begin
                result_input_valid = alpha_output_valid;
                result_input_last = 1'b1;
                alpha_output_ready = result_prestage_ready;
            end
            ST_VECTOR_START: vector_input_valid = 1'b1;
            ST_VECTOR_OUTPUT: begin
                result_input_valid = vector_output_valid;
                result_input_last = output_word_index == 3;
                vector_output_ready = result_prestage_ready
                    && output_word_index == 3;
            end
            ST_LAYERNORM_START: ln_start_valid = 1'b1;
            ST_LAYERNORM_LOAD0, ST_LAYERNORM_LOAD1: s_axis_tready = 1'b1;
            ST_LAYERNORM_SEND: ln_sample_valid = 1'b1;
            ST_LAYERNORM_OUTPUT: begin
                result_input_valid = ln_result_valid;
                result_input_last = ln_result_last;
                ln_result_ready = result_prestage_ready;
            end
            ST_SOFTMAX_START: softmax_start_valid = 1'b1;
            ST_SOFTMAX_LOAD: begin
                softmax_score_valid = s_axis_tvalid;
                // Entry to this state is delayed until the child accepted its
                // start command, and the child remains ready until score_last.
                // Making that invariant explicit breaks a ready-to-state loop.
                s_axis_tready = softmax_score_ready;
            end
            ST_SOFTMAX_OUTPUT: begin
                result_input_valid = softmax_probability_valid;
                result_input_last = softmax_probability_last;
                softmax_probability_ready = result_prestage_ready;
            end
            ST_LIF_CLEAR: lif_clear_state = 1'b1;
            ST_LIF_START: lif_input_valid = 1'b1;
            ST_LIF_OUTPUT: begin
                result_input_valid = lif_output_valid;
                result_input_last = !lif_trace_requested_r
                    || output_word_index == 12;
                lif_output_ready = result_prestage_ready
                    && (!lif_trace_requested_r || output_word_index == 12);
            end
            ST_OPTIMIZER_START: optimizer_input_valid = 1'b1;
            ST_OPTIMIZER_OUTPUT: begin
                result_input_valid = optimizer_output_valid;
                result_input_last = output_word_index == 4'd7;
                optimizer_output_ready = result_prestage_ready
                    && output_word_index == 4'd7;
            end
            ST_TRAIN_START: train_start_valid = 1'b1;
            ST_TRAIN_RUN: begin
                train_s_valid = s_axis_tvalid;
                s_axis_tready = train_s_ready;
                result_input_valid = train_m_valid;
                result_input_last = train_m_last;
                train_m_ready = result_prestage_ready;
            end
            ST_LUT_LOAD: begin
                lut_load_valid = lut_load_pending_r;
                s_axis_tready = !lut_load_pending_r;
            end
            ST_LUT_EVAL_START: lut_input_valid = 1'b1;
            ST_LUT_EVAL_OUTPUT: begin
                result_input_valid = lut_output_valid;
                result_input_last = output_word_index == 1;
                lut_output_ready = result_prestage_ready
                    && output_word_index == 1;
            end
            ST_LUT_CLEAR: lut_clear = lut_clear_requested_r;
            default: begin end
        endcase
    end

    // A bank samples whenever its source owns an available prestage slot.  The
    // write condition deliberately excludes child VALID: during a downstream
    // stall the full prestage freezes the bank, while an available slot may
    // harmlessly sample an invalid payload.  This keeps the wide bank enable
    // local to the one-hot core state instead of broadcasting a remote child
    // FSM bit, and it preserves the payload associated with every valid token.
    always_ff @(posedge clk) begin
        if (result_prestage_ready) begin
            unique case (state)
                ST_TENSOR: result_tensor_data_r <= tensor_m_data;
                ST_ALPHA_OUTPUT: result_alpha_data_r <= {28'd0,
                    alpha_zero_matches, alpha_spike_matches, alpha_score};
                ST_VECTOR_OUTPUT: result_vector_data_r
                    <= vector_result[output_word_index*64 +: 64];
                ST_LAYERNORM_OUTPUT: result_layernorm_data_r <= {19'd0,
                    ln_result_last, ln_result_index, ln_result_value};
                ST_SOFTMAX_OUTPUT: result_softmax_data_r <= {38'd0,
                    softmax_probability_last, softmax_probability_index,
                    softmax_probability};
                ST_LIF_OUTPUT: begin
                    if (output_word_index == 0) begin
                        result_lif_data_r <= {lif_timesteps, lif_spikes};
                    end else begin
                        result_lif_data_r <= lif_pre_reset_membrane[
                            lif_trace_bit_offset +: 64];
                    end
                end
                ST_OPTIMIZER_OUTPUT: result_optimizer_data_r <= {24'd0,
                    optimizer_updated_velocity[output_word_index*24 +: 24],
                    optimizer_updated_weight[output_word_index*16 +: 16]};
                ST_TRAIN_RUN: result_train_data_r <= train_m_data;
                ST_LUT_EVAL_OUTPUT: result_lut_data_r
                    <= lut_output_values[output_word_index*64 +: 64];
                default: begin end
            endcase
        end
    end

    // Two-entry elastic output pipeline.  The first stage captures each
    // engine into a source-local register; the second stage performs only the
    // registered-bank selection.  Data and tag registers intentionally have
    // no reset mux because their valid bits are the architectural qualifier.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result_prestage_valid_r <= 1'b0;
            result_output_valid_r <= 1'b0;
        end else begin
            if (result_output_ready) begin
                result_output_valid_r <= result_prestage_valid_r;
                if (result_prestage_valid_r) begin
                    result_output_last_r <= result_prestage_last_r;
                    unique case (result_prestage_source_r)
                        RESULT_TENSOR: result_output_data_r <= result_tensor_data_r;
                        RESULT_ALPHA: result_output_data_r <= result_alpha_data_r;
                        RESULT_VECTOR: result_output_data_r <= result_vector_data_r;
                        RESULT_LAYERNORM: result_output_data_r
                            <= result_layernorm_data_r;
                        RESULT_SOFTMAX: result_output_data_r <= result_softmax_data_r;
                        RESULT_LIF: result_output_data_r <= result_lif_data_r;
                        RESULT_OPTIMIZER: result_output_data_r
                            <= result_optimizer_data_r;
                        RESULT_TRAIN: result_output_data_r <= result_train_data_r;
                        RESULT_LUT: result_output_data_r <= result_lut_data_r;
                        default: result_output_data_r <= 64'd0;
                    endcase
                end
            end

            if (result_prestage_ready) begin
                result_prestage_valid_r <= result_input_valid;
                result_prestage_last_r <= result_input_last;
                unique case (state)
                    ST_TENSOR: result_prestage_source_r <= RESULT_TENSOR;
                    ST_ALPHA_OUTPUT: result_prestage_source_r <= RESULT_ALPHA;
                    ST_VECTOR_OUTPUT: result_prestage_source_r <= RESULT_VECTOR;
                    ST_LAYERNORM_OUTPUT: result_prestage_source_r
                        <= RESULT_LAYERNORM;
                    ST_SOFTMAX_OUTPUT: result_prestage_source_r <= RESULT_SOFTMAX;
                    ST_LIF_OUTPUT: result_prestage_source_r <= RESULT_LIF;
                    ST_OPTIMIZER_OUTPUT: result_prestage_source_r
                        <= RESULT_OPTIMIZER;
                    ST_TRAIN_RUN: result_prestage_source_r <= RESULT_TRAIN;
                    ST_LUT_EVAL_OUTPUT: result_prestage_source_r <= RESULT_LUT;
                    default: result_prestage_source_r <= RESULT_TENSOR;
                endcase
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            command_done <= 1'b0;
            command_done_pending_r <= 1'b0;
            input_beat_accepted_r <= 1'b0;
            output_beat_accepted_r <= 1'b0;
            status_error_r <= 1'b0;
            error_event_r <= 1'b0;
            clear_error_r <= 1'b0;
            stream_protocol_error_event_r <= 1'b0;
            clear_counters_event_r <= 1'b0;
            status_active_opcode <= 4'd0;
            status_command_count <= 32'd0;
            status_input_beats <= 32'd0;
            status_output_beats <= 32'd0;
            stream_beat_index <= 8'd0;
            stream_item_index <= 12'd0;
            stream_items_remaining_r <= 12'd0;
            stream_last_pending_r <= 1'b0;
            output_word_index <= 4'd0;
            captured_command_r <= 109'd0;
            train_reserved_partial_r <= 4'd0;
            train_reserved_nonzero_r <= 1'b0;
            command_had_output_r <= 1'b0;
            final_output_accepted_r <= 1'b0;
            completion_error_r <= 1'b0;
            alpha_query_r <= '0;
            alpha_key_r <= '0;
            alpha_active_bits_r <= 9'd0;
            alpha_q8_r <= 9'd0;
            alpha_load_valid_r <= 1'b0;
            alpha_load_index_r <= 3'd0;
            alpha_load_data_r <= 64'd0;
            alpha_load_final_r <= 1'b0;
            alpha_load_last_r <= 1'b0;
            vector_operation_r <= 4'd0;
            vector_operand_a_r <= '0;
            vector_operand_b_r <= '0;
            vector_multiplier_r <= 16'sd0;
            vector_shift_r <= 5'd0;
            vector_zero_point_r <= 16'sd0;
            vector_threshold_r <= 32'sd0;
            vector_window_r <= 32'd0;
            vector_clamp_min_r <= 32'sh80000000;
            vector_clamp_max_r <= 32'sh7fffffff;
            vector_load_valid_r <= 1'b0;
            vector_load_index_r <= 4'd0;
            vector_load_data_r <= 64'd0;
            vector_load_final_r <= 1'b0;
            vector_load_last_r <= 1'b0;
            ln_count_r <= 12'd0;
            ln_epsilon_r <= 32'd0;
            ln_sample_word0_r <= 64'd0;
            ln_sample_value_r <= 24'sd0;
            ln_sample_gamma_r <= 24'sd0;
            ln_sample_beta_r <= 24'sd0;
            ln_start_accept_r <= 1'b0;
            softmax_count_r <= 9'd0;
            softmax_shift_r <= 5'd0;
            softmax_start_accept_r <= 1'b0;
            lif_current_r <= '0;
            lif_leak_shift_r <= 5'd0;
            lif_threshold_r <= 24'sd0;
            lif_reset_r <= 24'sd0;
            lif_clear_requested_r <= 1'b0;
            lif_trace_requested_r <= 1'b0;
            lif_load_word_select_r <= 8'd0;
            lif_load_pending_r <= 1'b0;
            lif_load_index_r <= 3'd0;
            lif_load_data_r <= 64'd0;
            lif_load_last_r <= 1'b0;
            lif_load_final_r <= 1'b0;
            optimizer_weight_r <= '0;
            optimizer_gradient_r <= '0;
            optimizer_velocity_r <= '0;
            optimizer_lr_r <= 18'd0;
            optimizer_momentum_r <= 17'd0;
            optimizer_decay_r <= 17'd0;
            optimizer_update_shift_r <= 5'd16;
            optimizer_load_data_r <= 64'd0;
            optimizer_load_last_r <= 1'b0;
            optimizer_load_index_r <= 3'd0;
            optimizer_load_final_r <= 1'b0;
            optimizer_commit_valid_r <= 1'b0;
            lut_load_pending_r <= 1'b0;
            lut_load_address_r <= 8'd0;
            lut_load_value_r <= 16'sd0;
            lut_load_last_r <= 1'b0;
            lut_load_final_r <= 1'b0;
            lut_indices_r <= 64'd0;
            lut_clear_requested_r <= 1'b0;
            decode_error_pending_r <= 1'b0;
        end else begin
            if (clear_error_r) status_error_r <= 1'b0;
            else if (error_event_r) status_error_r <= 1'b1;
            error_event_r <= 1'b0;
            clear_error_r <= 1'b0;
            if (stream_protocol_error_event_r) error_event_r <= 1'b1;
            stream_protocol_error_event_r <= 1'b0;
            clear_counters_event_r <= 1'b0;
            vector_load_valid_r <= 1'b0;
            command_done <= command_done_pending_r;
            command_done_pending_r <= 1'b0;
            optimizer_commit_valid_r <= 1'b0;
            input_beat_accepted_r <= s_axis_tvalid && s_axis_tready;
            output_beat_accepted_r <= m_axis_tvalid && m_axis_tready;
            softmax_start_accept_r <= softmax_start_valid && softmax_start_ready;
            ln_start_accept_r <= ln_start_valid && ln_start_ready;
            if (result_input_accept) command_had_output_r <= 1'b1;
            if (result_output_accept && result_output_last_r) begin
                final_output_accepted_r <= 1'b1;
            end
            lif_load_word_select_r <= 8'd0;
            lif_load_pending_r <= 1'b0;
            for (int lif_word = 0; lif_word < 8; lif_word++) begin
                if (lif_load_word_select_r[lif_word]) begin
                    lif_current_r[lif_word*64 +: 64] <= lif_load_data_r;
                end
            end
            alpha_load_valid_r <= 1'b0;
            if (alpha_load_valid_r) begin
                if (alpha_load_last_r != alpha_load_final_r) begin
                    stream_protocol_error_event_r <= 1'b1;
                end
                if (alpha_load_index_r < 4) begin
                    alpha_query_r[alpha_load_index_r*64 +: 64] <= alpha_load_data_r;
                end else begin
                    alpha_key_r[(int'(alpha_load_index_r)-4)*64 +: 64]
                        <= alpha_load_data_r;
                end
            end
            if (optimizer_commit_valid_r) begin
                optimizer_weight_r[optimizer_load_index_r*16 +: 16]
                    <= optimizer_load_data_r[15:0];
                optimizer_gradient_r[optimizer_load_index_r*24 +: 24]
                    <= optimizer_load_data_r[39:16];
                optimizer_velocity_r[optimizer_load_index_r*24 +: 24]
                    <= optimizer_load_data_r[63:40];
            end
            if (vector_load_valid_r) begin
                if (vector_load_last_r != vector_load_final_r) begin
                    stream_protocol_error_event_r <= 1'b1;
                end
                if (vector_load_index_r == 0) begin
                    vector_clamp_min_r <= vector_load_data_r[31:0];
                    vector_clamp_max_r <= vector_load_data_r[63:32];
                end else if (vector_load_index_r <= 4) begin
                    vector_operand_a_r[(int'(vector_load_index_r)-1)*64 +: 64]
                        <= vector_load_data_r;
                end else begin
                    vector_operand_b_r[(int'(vector_load_index_r)-5)*64 +: 64]
                        <= vector_load_data_r;
                end
            end
            if (clear_counters_event_r) begin
                status_input_beats <= 32'd0;
                status_output_beats <= 32'd0;
            end else begin
                if (input_beat_accepted_r) begin
                    status_input_beats <= status_input_beats + 1'b1;
                end
                if (output_beat_accepted_r) begin
                    status_output_beats <= status_output_beats + 1'b1;
                end
            end

            unique case (state)
                ST_IDLE: begin
                    if (command_valid && command_ready) begin
                        captured_command_r <= command_data[108:0];
                        // Stage the wide reserved-field reduction across the
                        // existing capture/decode clocks for 200 MHz timing.
                        train_reserved_partial_r[0] <= |command_data[61:30];
                        train_reserved_partial_r[1] <= |command_data[93:62];
                        train_reserved_partial_r[2] <= |command_data[125:94];
                        train_reserved_partial_r[3] <= |command_data[127:126];
                        status_active_opcode <= command_data[3:0];
                        status_command_count <= status_command_count + 1'b1;
                        stream_beat_index <= 8'd0;
                        stream_item_index <= 12'd0;
                        stream_items_remaining_r <= 12'd0;
                        stream_last_pending_r <= 1'b0;
                        output_word_index <= 4'd0;
                        command_had_output_r <= 1'b0;
                        final_output_accepted_r <= 1'b0;
                        completion_error_r <= 1'b0;
                        state <= ST_DECODE_COMMAND;
                    end
                end

                ST_DECODE_COMMAND: begin
                    unique case (captured_command_r[3:0])
                    CMD_TRAIN_LINEAR_BACKWARD: begin
                        train_reserved_nonzero_r <= |train_reserved_partial_r;
                        state <= ST_TRAIN_START;
                    end
                    CMD_LOAD_ACTIVATIONS, CMD_LOAD_WEIGHTS_OUTER,
                    CMD_LOAD_WEIGHTS_DOT8, CMD_COMPUTE_DENSE_OUTER,
                    CMD_COMPUTE_DENSE_DOT8, CMD_COMPUTE_BINARY,
                    CMD_COMPUTE_TERNARY: begin
                        if (tensor_command_ready) state <= ST_TENSOR;
                    end
                    CMD_ALPHA_XNOR: begin
                        alpha_active_bits_r <= captured_command_r[50:42];
                        alpha_q8_r <= captured_command_r[59:51];
                        // ST_ALPHA_LOAD commits all four query words and all
                        // four key words before start, so retaining the prior
                        // command here cannot expose stale operand bits.
                        state <= ST_ALPHA_LOAD;
                    end
                    CMD_VECTOR_ALU: begin
                        vector_operation_r <= captured_command_r[7:4];
                        vector_multiplier_r <= captured_command_r[23:8];
                        vector_shift_r <= captured_command_r[28:24];
                        vector_zero_point_r <= captured_command_r[44:29];
                        vector_threshold_r <= captured_command_r[76:45];
                        vector_window_r <= captured_command_r[108:77];
                        state <= ST_VECTOR_LOAD;
                    end
                    CMD_LAYERNORM: begin
                        ln_count_r <= captured_command_r[15:4];
                        stream_items_remaining_r <= captured_command_r[15:4];
                        stream_last_pending_r <= captured_command_r[15:4] == 1;
                        ln_epsilon_r <= captured_command_r[47:16];
                        state <= ST_LAYERNORM_START;
                    end
                    CMD_LOG_SOFTMAX: begin
                        softmax_count_r <= captured_command_r[12:4];
                        stream_items_remaining_r <= {3'd0, captured_command_r[12:4]};
                        stream_last_pending_r <= captured_command_r[12:4] == 1;
                        softmax_shift_r <= captured_command_r[17:13];
                        state <= ST_SOFTMAX_START;
                    end
                    CMD_LIF_STEP: begin
                        lif_leak_shift_r <= captured_command_r[8:4];
                        lif_threshold_r <= captured_command_r[32:9];
                        lif_reset_r <= captured_command_r[56:33];
                        lif_clear_requested_r <= captured_command_r[57];
                        lif_trace_requested_r <= captured_command_r[58];
                        // The protocol always loads all eight words before
                        // starting the child, so a 512-bit command-time clear
                        // would only add a global mux to every current bit.
                        lif_load_word_select_r <= 8'd0;
                        lif_load_pending_r <= 1'b0;
                        lif_load_index_r <= 3'd0;
                        lif_load_final_r <= 1'b0;
                        state <= ST_LIF_LOAD;
                    end
                    CMD_SGD_UPDATE: begin
                        optimizer_lr_r <= captured_command_r[21:4];
                        optimizer_momentum_r <= captured_command_r[38:22];
                        optimizer_decay_r <= captured_command_r[55:39];
                        optimizer_update_shift_r <= captured_command_r[60:56];
                        state <= ST_OPTIMIZER_LOAD;
                    end
                    CMD_ACTIVATION_LUT: begin
                        stream_items_remaining_r <= captured_command_r[19:8];
                        stream_last_pending_r <= captured_command_r[19:8] == 1;
                        if (captured_command_r[7:4] == 0) begin
                            lut_clear_requested_r <= 1'b1;
                            decode_error_pending_r <= 1'b0;
                            state <= ST_LUT_CLEAR;
                        end else if (captured_command_r[7:4] == 1
                                && captured_command_r[19:8] != 0) begin
                            lut_load_pending_r <= 1'b0;
                            state <= ST_LUT_LOAD;
                        end else if (captured_command_r[7:4] == 2) begin
                            state <= ST_LUT_EVAL_LOAD;
                        end else begin
                            lut_clear_requested_r <= 1'b0;
                            decode_error_pending_r <= 1'b1;
                            state <= ST_LUT_CLEAR;
                        end
                    end
                    CMD_CLEAR_STATUS: begin
                        if (tensor_command_ready) begin
                            clear_error_r <= 1'b1;
                            clear_counters_event_r <= 1'b1;
                            state <= ST_TENSOR;
                        end
                    end
                    default: begin
                        lut_clear_requested_r <= 1'b0;
                        decode_error_pending_r <= 1'b1;
                        state <= ST_LUT_CLEAR;
                    end
                    endcase
                end

                    ST_TENSOR: begin
                        if (tensor_command_done) begin
                            if (command_had_output_r || tensor_result_accept) begin
                                completion_error_r <= tensor_status_error;
                                state <= ST_OUTPUT_DRAIN;
                            end else begin
                                if (tensor_status_error) error_event_r <= 1'b1;
                                command_done_pending_r <= 1'b1;
                                state <= ST_IDLE;
                            end
                        end
                    end
                    ST_ALPHA_LOAD: begin
                        if (alpha_load_valid_r && alpha_load_final_r) begin
                            alpha_load_final_r <= 1'b0;
                            state <= ST_ALPHA_DRAIN;
                        end else if (s_axis_tvalid) begin
                            alpha_load_valid_r <= 1'b1;
                            alpha_load_index_r <= stream_beat_index[2:0];
                            alpha_load_data_r <= s_axis_tdata;
                            alpha_load_final_r <= stream_beat_index == 7;
                            alpha_load_last_r <= s_axis_tlast;
                            if (stream_beat_index != 7) begin
                                stream_beat_index <= stream_beat_index + 1'b1;
                            end
                        end
                    end
                    ST_ALPHA_DRAIN: if (!alpha_load_valid_r) state <= ST_ALPHA_START;
                    ST_ALPHA_START: if (alpha_input_ready) state <= ST_ALPHA_OUTPUT;
                    ST_ALPHA_OUTPUT: if (alpha_result_accept) begin
                        completion_error_r <= alpha_config_error;
                        state <= ST_OUTPUT_DRAIN;
                    end
                    ST_VECTOR_LOAD: begin
                        if (vector_load_final_r) begin
                            vector_load_final_r <= 1'b0;
                            state <= ST_VECTOR_DRAIN;
                        end else if (s_axis_tvalid) begin
                            vector_load_valid_r <= 1'b1;
                            vector_load_index_r <= stream_beat_index[3:0];
                            vector_load_data_r <= s_axis_tdata;
                            vector_load_final_r <= stream_beat_index == 8;
                            vector_load_last_r <= s_axis_tlast;
                            if (stream_beat_index != 8) begin
                                stream_beat_index <= stream_beat_index + 1'b1;
                            end
                        end
                    end
                    // The registered final payload commits at this edge.
                    ST_VECTOR_DRAIN: state <= ST_VECTOR_START;
                    ST_VECTOR_START: if (vector_input_ready) begin
                        output_word_index <= 4'd0;
                        state <= ST_VECTOR_OUTPUT;
                    end
                    ST_VECTOR_OUTPUT: if (vector_result_accept) begin
                        if (output_word_index == 3) begin
                            completion_error_r <= vector_config_error;
                            state <= ST_OUTPUT_DRAIN;
                        end else output_word_index <= output_word_index + 1'b1;
                    end
                    ST_LAYERNORM_START: if (ln_start_accept_r) state <= ST_LAYERNORM_LOAD0;
                    ST_LAYERNORM_LOAD0: if (s_axis_tvalid) begin
                        ln_sample_word0_r <= s_axis_tdata;
                        state <= ST_LAYERNORM_LOAD1;
                    end
                    ST_LAYERNORM_LOAD1: if (s_axis_tvalid) begin
                        ln_sample_value_r <= ln_sample_word0_r[23:0];
                        ln_sample_gamma_r <= ln_sample_word0_r[47:24];
                        ln_sample_beta_r <= {s_axis_tdata[7:0], ln_sample_word0_r[63:48]};
                        if (s_axis_tlast != stream_last_pending_r) begin
                            error_event_r <= 1'b1;
                        end
                        state <= ST_LAYERNORM_SEND;
                    end
                    ST_LAYERNORM_SEND: if (ln_sample_ready) begin
                        if (stream_last_pending_r) begin
                            state <= ST_LAYERNORM_OUTPUT;
                        end else begin
                            stream_item_index <= stream_item_index + 1'b1;
                            stream_items_remaining_r <= stream_items_remaining_r - 1'b1;
                            stream_last_pending_r <= stream_items_remaining_r == 12'd2;
                            state <= ST_LAYERNORM_LOAD0;
                        end
                    end
                    ST_LAYERNORM_OUTPUT: if (layernorm_result_accept
                            && ln_result_last) begin
                        completion_error_r <= ln_config_error;
                        state <= ST_OUTPUT_DRAIN;
                    end
                    ST_SOFTMAX_START: if (softmax_start_accept_r) state <= ST_SOFTMAX_LOAD;
                    ST_SOFTMAX_LOAD: if (s_axis_tvalid) begin
                        if (s_axis_tlast != stream_last_pending_r) begin
                            error_event_r <= 1'b1;
                        end
                        if (stream_last_pending_r) begin
                            state <= ST_SOFTMAX_OUTPUT;
                        end else begin
                            stream_item_index <= stream_item_index + 1'b1;
                            stream_items_remaining_r <= stream_items_remaining_r - 1'b1;
                            stream_last_pending_r <= stream_items_remaining_r == 12'd2;
                        end
                    end
                    ST_SOFTMAX_OUTPUT: if (softmax_result_accept
                            && softmax_probability_last) begin
                        completion_error_r <= softmax_config_error;
                        state <= ST_OUTPUT_DRAIN;
                    end
                    ST_LIF_LOAD: begin
                        if (lif_load_pending_r) begin
                            if (lif_load_last_r != lif_load_final_r) begin
                                error_event_r <= 1'b1;
                            end
                            if (lif_load_final_r) begin
                                if (lif_clear_requested_r) state <= ST_LIF_CLEAR;
                                else state <= ST_LIF_START;
                            end else begin
                                lif_load_index_r <= lif_load_index_r + 1'b1;
                            end
                        end else if (s_axis_tvalid) begin
                            lif_load_pending_r <= 1'b1;
                            lif_load_word_select_r <= 8'b1 << lif_load_index_r;
                            lif_load_data_r <= s_axis_tdata;
                            lif_load_last_r <= s_axis_tlast;
                            lif_load_final_r <= lif_load_index_r == 7;
                        end
                    end
                    ST_LIF_CLEAR: state <= ST_LIF_START;
                    ST_LIF_START: if (lif_input_ready) begin
                        output_word_index <= 4'd0;
                        state <= ST_LIF_OUTPUT;
                    end
                    ST_LIF_OUTPUT: if (lif_result_accept) begin
                        if (!lif_trace_requested_r || output_word_index == 12) begin
                            completion_error_r <= 1'b0;
                            state <= ST_OUTPUT_DRAIN;
                        end else begin
                            output_word_index <= output_word_index + 1'b1;
                        end
                    end
                    ST_OPTIMIZER_LOAD: if (s_axis_tvalid) begin
                        optimizer_load_data_r <= s_axis_tdata;
                        optimizer_load_last_r <= s_axis_tlast;
                        optimizer_load_index_r <= stream_beat_index[2:0];
                        optimizer_load_final_r <= stream_beat_index == 7;
                        optimizer_commit_valid_r <= 1'b1;
                        state <= ST_OPTIMIZER_COMMIT;
                    end
                    ST_OPTIMIZER_COMMIT: begin
                        if (optimizer_load_last_r != optimizer_load_final_r) begin
                            error_event_r <= 1'b1;
                        end
                        if (optimizer_load_final_r) begin
                            state <= ST_OPTIMIZER_START;
                        end else begin
                            stream_beat_index <= stream_beat_index + 1'b1;
                            state <= ST_OPTIMIZER_LOAD;
                        end
                    end
                    ST_OPTIMIZER_START: if (optimizer_input_ready) begin
                        output_word_index <= 4'd0;
                        state <= ST_OPTIMIZER_OUTPUT;
                    end
                    ST_OPTIMIZER_OUTPUT: if (optimizer_result_accept) begin
                        if (output_word_index == 4'd7) begin
                            completion_error_r <= optimizer_config_error;
                            state <= ST_OUTPUT_DRAIN;
                        end else output_word_index <= output_word_index + 1'b1;
                    end
                    ST_TRAIN_START: if (train_start_ready) state <= ST_TRAIN_RUN;
                    ST_TRAIN_RUN: if (train_done) begin
                        if (command_had_output_r || train_result_accept) begin
                            completion_error_r <= train_config_error;
                            state <= ST_OUTPUT_DRAIN;
                        end else begin
                            if (train_config_error) error_event_r <= 1'b1;
                            command_done_pending_r <= 1'b1;
                            state <= ST_IDLE;
                        end
                    end
                    ST_LUT_LOAD: begin
                        if (lut_load_pending_r && lut_load_ready) begin
                            lut_load_pending_r <= 1'b0;
                            if (lut_load_last_r != lut_load_final_r) begin
                                error_event_r <= 1'b1;
                            end
                            if (lut_load_final_r) begin
                                command_done_pending_r <= 1'b1;
                                state <= ST_IDLE;
                            end else begin
                                stream_item_index <= stream_item_index + 1'b1;
                                stream_items_remaining_r
                                    <= stream_items_remaining_r - 1'b1;
                                stream_last_pending_r
                                    <= stream_items_remaining_r == 12'd2;
                            end
                        end else if (!lut_load_pending_r && s_axis_tvalid) begin
                            lut_load_pending_r <= 1'b1;
                            lut_load_address_r <= s_axis_tdata[7:0];
                            lut_load_value_r <= s_axis_tdata[23:8];
                            lut_load_last_r <= s_axis_tlast;
                            lut_load_final_r <= stream_last_pending_r;
                        end
                    end
                    ST_LUT_EVAL_LOAD: if (s_axis_tvalid) begin
                        lut_indices_r <= s_axis_tdata;
                        if (!s_axis_tlast) error_event_r <= 1'b1;
                        state <= ST_LUT_EVAL_START;
                    end
                    ST_LUT_EVAL_START: if (lut_input_ready) begin
                        output_word_index <= 4'd0;
                        state <= ST_LUT_EVAL_OUTPUT;
                    end
                    ST_LUT_EVAL_OUTPUT: begin
                        if (lut_result_accept && output_word_index == 1) begin
                            completion_error_r <= lut_config_error;
                            state <= ST_OUTPUT_DRAIN;
                        end else if (lut_result_accept) begin
                            output_word_index <= output_word_index + 1'b1;
                        end
                    end
                    ST_OUTPUT_DRAIN: if (final_output_accepted_r) begin
                        if (completion_error_r) error_event_r <= 1'b1;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end
                    ST_LUT_CLEAR: begin
                        if (decode_error_pending_r) error_event_r <= 1'b1;
                        lut_clear_requested_r <= 1'b0;
                        decode_error_pending_r <= 1'b0;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end
                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
