`timescale 1ns / 1ps
`default_nettype none

// Exact tiled reverse-mode engine for Y = XW:
//   dX = dY * transpose(W), dW = transpose(X) * dY.
// The external tile remains 8x8x8, while a 4x4 DSP array traverses two
// result-column groups and up to two reduction groups. One shared 47-bit
// accumulator bank preserves cross-command partial sums until the compiler
// requests the final rounded output.
module neuraedge_linear_backward8 #(
    parameter int LANES = 8
) (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_counters,

    input  wire start_valid,
    output logic start_ready,
    input  wire [3:0] active_rows,
    input  wire [3:0] active_inputs,
    input  wire [3:0] active_outputs,
    input  wire [4:0] input_gradient_shift,
    input  wire [4:0] weight_gradient_shift,
    input  wire [1:0] gradient_mode,
    input  wire accumulate_result,
    input  wire suppress_output,
    input  wire reserved_nonzero,

    input  wire s_axis_tvalid,
    output logic s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire s_axis_tlast,

    output logic m_axis_tvalid,
    input  wire m_axis_tready,
    output logic [63:0] m_axis_tdata,
    output logic m_axis_tlast,

    output logic busy,
    output logic done,
    output logic config_error,
    (* extract_enable = "no" *)
    output logic [31:0] completed_tiles,
    output logic [31:0] active_cycles
);
    localparam int RESULT_LANES = 4;
    localparam int REDUCTION_LANES = 4;

    localparam logic [1:0] MODE_BOTH = 2'd0;
    localparam logic [1:0] MODE_INPUT_GRADIENT = 2'd1;
    localparam logic [1:0] MODE_WEIGHT_GRADIENT = 2'd2;

    typedef enum logic [4:0] {
        ST_IDLE = 5'd0,
        ST_LOAD_X = 5'd1,
        ST_LOAD_W = 5'd2,
        ST_LOAD_DY = 5'd3,
        ST_MULTIPLY = 5'd4,
        ST_PRODUCT = 5'd5,
        ST_PAIR = 5'd6,
        ST_REDUCE = 5'd7,
        ST_COMMIT = 5'd8,
        ST_OUTPUT_DX = 5'd9,
        ST_OUTPUT_DW = 5'd10,
        ST_SWITCH_DW = 5'd11,
        ST_DONE = 5'd12,
        ST_ROUND_MAGNITUDE = 5'd13,
        ST_ROUND_BIAS = 5'd14,
        ST_ROUND_SHIFT = 5'd15,
        ST_ROUND_SIGN = 5'd16,
        ST_SATURATE = 5'd17,
        ST_ADD_ACCUMULATOR = 5'd18,
        ST_STORE_ACCUMULATOR = 5'd19,
        ST_REDUCTION_SUM = 5'd20,
        ST_STORE_RESULT = 5'd21,
        ST_REGISTER_OPERANDS = 5'd22,
        ST_CONFIGURE = 5'd23,
        ST_ROUND_MAGNITUDE_HIGH = 5'd24,
        ST_ROUND_BIAS_HIGH = 5'd25,
        ST_ROUND_SIGN_HIGH = 5'd26,
        ST_LOAD_COMMIT_DRAIN = 5'd27
    } state_t;

    // Keep the compact encoding, but route next-state decisions to D rather
    // than the tighter-setup CE pin used by automatic enable extraction.
    (* fsm_encoding = "sequential", extract_enable = "no" *) state_t state;
    logic [3:0] rows_r;
    logic [3:0] inputs_r;
    logic [3:0] outputs_r;
    logic [4:0] dw_shift_r;
    logic [4:0] result_shift_r;
    logic [46:0] rounding_bias_r;
    logic [1:0] mode_r;
    logic [1:0] compute_mode_r;
    logic suppress_output_r;
    logic accumulate_result_r;
    logic start_configuration_invalid_r;
    logic [1:0] requested_accumulator_mode_r;
    logic [3:0] requested_accumulator_rows_r;
    logic [3:0] requested_accumulator_columns_r;
    logic [3:0] compute_result_row_count_r;
    logic [3:0] compute_result_column_count_r;
    logic [3:0] compute_reduction_count_r;

    // Each lane owns a preserved copy of the mode-select bit. This limits the
    // physical fanout into the four DSP lanes while retaining one logical mode.
    (* keep = "true", dont_touch = "true" *)
    logic [RESULT_LANES-1:0] input_gradient_lane_r;

    // These banks are loaded under several FSM states.  Implementing the hold
    // condition on D keeps a distant state decode off thousands of FDRE CE
    // pins and lets Vivado place each small input mux beside its register.
    (* extract_enable = "no" *)
    logic signed [15:0] activation_r [0:LANES-1][0:LANES-1];
    (* extract_enable = "no" *)
    logic signed [15:0] weight_r [0:LANES-1][0:LANES-1];
    (* extract_enable = "no" *)
    logic signed [15:0] output_gradient_r [0:LANES-1][0:LANES-1];

    // Four banks provide the four independent reads and writes needed by the
    // result lanes. Each bank stores one column from both four-column groups
    // for every row: address = {row, column_group}. Commands that do not
    // accumulate select mathematical zero instead of clearing the RAM. This
    // preserves exact accumulation while removing a clear/hold/store mux and
    // its FSM control route from 3,008 individual flip-flops.
    wire signed [46:0] accumulator_bank_read_data
        [0:RESULT_LANES-1];
    wire signed [46:0] accumulator_bank_write_data
        [0:RESULT_LANES-1];
    wire [RESULT_LANES-1:0] accumulator_bank_write_enable;
    logic accumulator_valid_r;
    logic [1:0] accumulator_mode_r;
    logic [3:0] accumulator_rows_r;
    logic [3:0] accumulator_columns_r;

    logic signed [31:0] result_row_buffer_r [0:LANES-1];
    // Eight signed 16x16 products fit exactly in 35 bits. Keeping this
    // command-local sum at its mathematical width shortens the carry chain;
    // it is sign-extended only when added to the retained 47-bit accumulator.
    logic signed [34:0] row_partial_r [0:RESULT_LANES-1];
    logic signed [33:0] reduction_sum_r [0:RESULT_LANES-1];
    logic signed [46:0] selected_accumulator_r [0:RESULT_LANES-1];
    logic signed [47:0] accumulated_wide_r [0:RESULT_LANES-1];
    logic commit_lane_valid_r [0:RESULT_LANES-1];
    logic [46:0] rounding_value_r [0:RESULT_LANES-1];
    logic rounding_negative_r [0:RESULT_LANES-1];
    logic rounding_carry_r [0:RESULT_LANES-1];
    logic signed [31:0] saturated_result_r [0:RESULT_LANES-1];

    logic [3:0] load_row_r;
    logic load_half_r;
    logic load_final_r;
    // Stream VALID previously participated directly in the D-input mux for
    // all 3,072 operand-bank bits.  This elastic stage accepts one beat every
    // clock, while the row-local commit enables below terminate control near
    // the destination registers.  The payload stage can consume its old beat
    // and capture the next operand class on the same edge.
    logic input_stage_valid_r;
    logic [63:0] input_stage_data_r;
    logic input_stage_last_r;
    logic [63:0] load_commit_data_r;
    logic [1:0] activation_write_enable_r [0:LANES-1];
    logic [1:0] weight_write_enable_r [0:LANES-1];
    logic [1:0] output_gradient_write_enable_r [0:LANES-1];
    // Keep the row hold as a D-input mux.  A dedicated CE would put the
    // multi-state row-update decode on the FDRE CE pin, whose setup time is
    // tighter than the data input on 7-series devices.
    (* extract_enable = "no" *) logic [3:0] result_row_r;
    logic column_group_r;
    logic reduction_group_r;
    logic [1:0] output_word_r;

    wire [2:0] load_row_index = load_row_r[2:0];
    wire [2:0] result_row_index = result_row_r[2:0];
    wire [3:0] accumulator_address = {result_row_index, column_group_r};
    wire [3:0] result_row_count = compute_result_row_count_r;
    wire [3:0] result_column_count = compute_result_column_count_r;
    wire [3:0] reduction_count = compute_reduction_count_r;
    wire input_load_state = state == ST_LOAD_X || state == ST_LOAD_W
        || state == ST_LOAD_DY;
    wire input_stage_consume = input_stage_valid_r && input_load_state;
    wire input_stage_ready = !input_stage_valid_r || input_stage_consume;
    wire input_final_buffered = state == ST_LOAD_DY && input_stage_valid_r
        && load_half_r && load_final_r;
    wire [1:0] requested_accumulator_mode = gradient_mode == MODE_WEIGHT_GRADIENT
        ? MODE_WEIGHT_GRADIENT : MODE_INPUT_GRADIENT;
    wire [3:0] requested_accumulator_rows = gradient_mode == MODE_WEIGHT_GRADIENT
        ? active_inputs : active_rows;
    wire [3:0] requested_accumulator_columns = gradient_mode == MODE_WEIGHT_GRADIENT
        ? active_outputs : active_inputs;
    wire dimensions_invalid = active_rows == 0 || active_rows > 4'(LANES)
        || active_inputs == 0 || active_inputs > 4'(LANES)
        || active_outputs == 0 || active_outputs > 4'(LANES);
    wire accumulation_shape_invalid = accumulate_result && (
        gradient_mode == MODE_BOTH
        || !accumulator_valid_r
        || accumulator_mode_r != requested_accumulator_mode
        || accumulator_rows_r != requested_accumulator_rows
        || accumulator_columns_r != requested_accumulator_columns);
    wire start_configuration_invalid = reserved_nonzero || dimensions_invalid
        || gradient_mode == 2'd3 || accumulation_shape_invalid;

    logic signed [15:0] multiplier_left [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    logic signed [15:0] multiplier_right [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    (* keep = "true", dont_touch = "true" *)
    logic signed [15:0] multiplier_left_r [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    (* keep = "true", dont_touch = "true" *)
    logic signed [15:0] multiplier_right_r [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    logic signed [15:0] multiplier_left_dsp_r [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    logic signed [15:0] multiplier_right_dsp_r [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    (* use_dsp = "yes" *) wire signed [31:0]
        multiplier_product [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    logic signed [31:0] product_r [0:RESULT_LANES-1][0:REDUCTION_LANES-1];
    logic signed [32:0] pair_sum_r [0:RESULT_LANES-1][0:1];
    // This local pulse drives only the eight pair-reduction DSP registers.
    // Registering it near the array avoids routing the shared FSM decode to
    // DSP clock-enable pins, whose setup requirement is especially tight.
    logic pair_update_r;

    wire [3:0] result_column_index [0:RESULT_LANES-1];
    wire [3:0] reduction_index [0:REDUCTION_LANES-1];
    logic [191:0] packed_weight_gradient;

    function automatic logic signed [46:0] saturate_s47(
        input logic signed [47:0] value
    );
        begin
            if (value[47] == value[46]) saturate_s47 = value[46:0];
            else if (value[47]) saturate_s47 = {1'b1, {46{1'b0}}};
            else saturate_s47 = {1'b0, {46{1'b1}}};
        end
    endfunction

    function automatic logic [46:0] round_bias(
        input logic [4:0] shift
    );
        begin
            if (shift == 0) round_bias = 47'd0;
            else round_bias = 47'd1 << (shift - 1'b1);
        end
    endfunction

    function automatic logic signed [31:0] saturate_s32(
        input logic signed [46:0] value
    );
        begin
            // A signed value fits in 32 bits exactly when every discarded bit
            // is a copy of bit 31.  This is equivalent to range comparisons,
            // but uses equality/reduction logic instead of two carry chains.
            if (value[46:31] == {16{value[31]}}) begin
                saturate_s32 = value[31:0];
            end else if (value[46]) begin
                saturate_s32 = 32'sh80000000;
            end else begin
                saturate_s32 = 32'sh7fffffff;
            end
        end
    endfunction

    function automatic logic signed [23:0] saturate_s24(
        input logic signed [46:0] value
    );
        begin
            if (value[46:23] == {24{value[23]}}) begin
                saturate_s24 = value[23:0];
            end else if (value[46]) begin
                saturate_s24 = 24'sh800000;
            end else begin
                saturate_s24 = 24'sh7fffff;
            end
        end
    endfunction

    function automatic logic signed [31:0] saturate_s24_extended(
        input logic signed [46:0] value
    );
        logic signed [23:0] saturated;
        begin
            saturated = saturate_s24(value);
            saturate_s24_extended = {{8{saturated[23]}}, saturated};
        end
    endfunction

    for (genvar bank = 0; bank < RESULT_LANES; bank++) begin : g_accumulator_bank
        assign accumulator_bank_write_enable[bank] =
            state == ST_STORE_ACCUMULATOR && commit_lane_valid_r[bank];
        assign accumulator_bank_write_data[bank] =
            saturate_s47(accumulated_wide_r[bank]);

        neuraedge_gradient_accumulator_bank u_bank (
            .clk(clk),
            .write_enable(accumulator_bank_write_enable[bank]),
            .write_address(accumulator_address),
            .write_data(accumulator_bank_write_data[bank]),
            .read_address(accumulator_address),
            .read_data(accumulator_bank_read_data[bank])
        );
    end

    for (genvar result_lane = 0; result_lane < RESULT_LANES;
            result_lane++) begin : g_result_lane
        assign result_column_index[result_lane] =
            (column_group_r ? 4'd4 : 4'd0) + 4'(result_lane);
        for (genvar reduction_lane = 0;
                reduction_lane < REDUCTION_LANES; reduction_lane++) begin : g_multiplier
            assign multiplier_product[result_lane][reduction_lane] =
                multiplier_left_dsp_r[result_lane][reduction_lane]
                * multiplier_right_dsp_r[result_lane][reduction_lane];
        end
    end

    for (genvar reduction_lane = 0; reduction_lane < REDUCTION_LANES;
            reduction_lane++) begin : g_reduction_index
        assign reduction_index[reduction_lane] =
            (reduction_group_r ? 4'd4 : 4'd0) + 4'(reduction_lane);
    end

    always_comb begin
        for (int result_lane = 0; result_lane < RESULT_LANES;
                result_lane++) begin
            for (int reduction_lane = 0; reduction_lane < REDUCTION_LANES;
                    reduction_lane++) begin
                multiplier_left[result_lane][reduction_lane] = 16'sd0;
                multiplier_right[result_lane][reduction_lane] = 16'sd0;
                if (result_row_r < result_row_count
                        && result_column_index[result_lane] < result_column_count
                        && reduction_index[reduction_lane] < reduction_count) begin
                    if (input_gradient_lane_r[result_lane]) begin
                        multiplier_left[result_lane][reduction_lane] =
                            output_gradient_r[result_row_index]
                                [reduction_index[reduction_lane][2:0]];
                        multiplier_right[result_lane][reduction_lane] =
                            weight_r[result_column_index[result_lane][2:0]]
                                [reduction_index[reduction_lane][2:0]];
                    end else begin
                        multiplier_left[result_lane][reduction_lane] =
                            activation_r[reduction_index[reduction_lane][2:0]]
                                [result_row_index];
                        multiplier_right[result_lane][reduction_lane] =
                            output_gradient_r[reduction_index[reduction_lane][2:0]]
                                [result_column_index[result_lane][2:0]];
                    end
                end
            end
        end

        packed_weight_gradient = '0;
        for (int lane = 0; lane < LANES; lane++) begin
            packed_weight_gradient[lane*24 +: 24] =
                result_row_buffer_r[lane][23:0];
        end
    end

    always_comb begin
        start_ready = state == ST_IDLE;
        busy = state != ST_IDLE;
        done = state == ST_DONE;
        s_axis_tready = input_load_state && input_stage_ready
            && !input_final_buffered;
        m_axis_tvalid = state == ST_OUTPUT_DX || state == ST_OUTPUT_DW;
        m_axis_tdata = 64'd0;
        m_axis_tlast = 1'b0;
        // compute_mode_r is fixed before an output row starts.  Selecting data
        // with it avoids putting the large FSM state decode in the stream-data
        // path while preserving the same value whenever tvalid is asserted.
        if (compute_mode_r == MODE_INPUT_GRADIENT) begin
            m_axis_tdata = {
                result_row_buffer_r[output_word_r*2 + 1],
                result_row_buffer_r[output_word_r*2]
            };
            m_axis_tlast = mode_r == MODE_INPUT_GRADIENT
                && result_row_r == rows_r - 1'b1 && output_word_r == 3;
        end else begin
            m_axis_tdata = packed_weight_gradient[output_word_r*64 +: 64];
            m_axis_tlast = result_row_r == inputs_r - 1'b1
                && output_word_r == 2;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            rows_r <= 4'd0;
            inputs_r <= 4'd0;
            outputs_r <= 4'd0;
            dw_shift_r <= 5'd0;
            result_shift_r <= 5'd0;
            rounding_bias_r <= 47'd0;
            mode_r <= MODE_BOTH;
            compute_mode_r <= MODE_INPUT_GRADIENT;
            suppress_output_r <= 1'b0;
            accumulate_result_r <= 1'b0;
            start_configuration_invalid_r <= 1'b0;
            requested_accumulator_mode_r <= MODE_INPUT_GRADIENT;
            requested_accumulator_rows_r <= 4'd0;
            requested_accumulator_columns_r <= 4'd0;
            compute_result_row_count_r <= 4'd0;
            compute_result_column_count_r <= 4'd0;
            compute_reduction_count_r <= 4'd0;
            input_gradient_lane_r <= {RESULT_LANES{1'b1}};
            load_row_r <= 4'd0;
            load_half_r <= 1'b0;
            load_final_r <= 1'b0;
            input_stage_valid_r <= 1'b0;
            input_stage_data_r <= 64'd0;
            input_stage_last_r <= 1'b0;
            load_commit_data_r <= 64'd0;
            result_row_r <= 4'd0;
            column_group_r <= 1'b0;
            reduction_group_r <= 1'b0;
            output_word_r <= 2'd0;
            accumulator_valid_r <= 1'b0;
            accumulator_mode_r <= MODE_INPUT_GRADIENT;
            accumulator_rows_r <= 4'd0;
            accumulator_columns_r <= 4'd0;
            pair_update_r <= 1'b0;
            config_error <= 1'b0;
            completed_tiles <= 32'd0;
            active_cycles <= 32'd0;
            for (int row = 0; row < LANES; row++) begin
                result_row_buffer_r[row] <= 32'sd0;
                activation_write_enable_r[row] <= 2'b00;
                weight_write_enable_r[row] <= 2'b00;
                output_gradient_write_enable_r[row] <= 2'b00;
                for (int column = 0; column < LANES; column++) begin
                    activation_r[row][column] <= 16'sd0;
                    weight_r[row][column] <= 16'sd0;
                    output_gradient_r[row][column] <= 16'sd0;
                end
            end
            for (int result_lane = 0; result_lane < RESULT_LANES;
                    result_lane++) begin
                row_partial_r[result_lane] <= 35'sd0;
                reduction_sum_r[result_lane] <= 34'sd0;
                selected_accumulator_r[result_lane] <= 47'sd0;
                accumulated_wide_r[result_lane] <= 48'sd0;
                commit_lane_valid_r[result_lane] <= 1'b0;
                rounding_value_r[result_lane] <= 47'd0;
                rounding_negative_r[result_lane] <= 1'b0;
                rounding_carry_r[result_lane] <= 1'b0;
                saturated_result_r[result_lane] <= 32'sd0;
                for (int reduction_lane = 0;
                        reduction_lane < REDUCTION_LANES; reduction_lane++) begin
                    multiplier_left_r[result_lane][reduction_lane] <= 16'sd0;
                    multiplier_right_r[result_lane][reduction_lane] <= 16'sd0;
                    multiplier_left_dsp_r[result_lane][reduction_lane] <= 16'sd0;
                    multiplier_right_dsp_r[result_lane][reduction_lane] <= 16'sd0;
                    product_r[result_lane][reduction_lane] <= 32'sd0;
                end
                pair_sum_r[result_lane][0] <= 33'sd0;
                pair_sum_r[result_lane][1] <= 33'sd0;
            end
        end else begin
            if (state != ST_IDLE) active_cycles <= active_cycles + 1'b1;
            pair_update_r <= state == ST_PRODUCT;

            // The single elastic slot is fall-through only at clock edges: an
            // old buffered beat can be consumed while the next stream beat is
            // captured. On the final dY beat READY is withheld because there is
            // no following operand class to prefetch.
            if (input_stage_ready) begin
                input_stage_valid_r <= s_axis_tvalid && s_axis_tready;
                if (s_axis_tvalid && s_axis_tready) begin
                    input_stage_data_r <= s_axis_tdata;
                    input_stage_last_r <= s_axis_tlast;
                end
            end

            // Apply the previous beat's row-local commit and prepare one-hot
            // enables for the beat consumed this cycle.  Each enable drives
            // only four 16-bit destinations, replacing the former chip-wide
            // stream-VALID/state route into all three operand banks.
            for (int row = 0; row < LANES; row++) begin
                for (int half = 0; half < 2; half++) begin
                    if (activation_write_enable_r[row][half]) begin
                        for (int lane = 0; lane < 4; lane++) begin
                            activation_r[row][half*4 + lane]
                                <= load_commit_data_r[lane*16 +: 16];
                        end
                    end
                    if (weight_write_enable_r[row][half]) begin
                        for (int lane = 0; lane < 4; lane++) begin
                            weight_r[row][half*4 + lane]
                                <= load_commit_data_r[lane*16 +: 16];
                        end
                    end
                    if (output_gradient_write_enable_r[row][half]) begin
                        for (int lane = 0; lane < 4; lane++) begin
                            output_gradient_r[row][half*4 + lane]
                                <= load_commit_data_r[lane*16 +: 16];
                        end
                    end
                end
                activation_write_enable_r[row] <= 2'b00;
                weight_write_enable_r[row] <= 2'b00;
                output_gradient_write_enable_r[row] <= 2'b00;
            end
            if (input_stage_valid_r && input_load_state) begin
                load_commit_data_r <= input_stage_data_r;
                unique case (state)
                    ST_LOAD_X: activation_write_enable_r[load_row_index]
                        [load_half_r] <= 1'b1;
                    ST_LOAD_W: weight_write_enable_r[load_row_index]
                        [load_half_r] <= 1'b1;
                    ST_LOAD_DY: output_gradient_write_enable_r[load_row_index]
                        [load_half_r] <= 1'b1;
                    default: begin end
                endcase
            end

            if (pair_update_r) begin
                for (int result_lane = 0; result_lane < RESULT_LANES;
                        result_lane++) begin
                    pair_sum_r[result_lane][0] <=
                        {product_r[result_lane][0][31], product_r[result_lane][0]}
                        + {product_r[result_lane][1][31], product_r[result_lane][1]};
                    pair_sum_r[result_lane][1] <=
                        {product_r[result_lane][2][31], product_r[result_lane][2]}
                        + {product_r[result_lane][3][31], product_r[result_lane][3]};
                end
            end
            unique case (state)
                ST_IDLE: begin
                    if (start_valid) begin
                        rows_r <= active_rows;
                        inputs_r <= active_inputs;
                        outputs_r <= active_outputs;
                        dw_shift_r <= weight_gradient_shift;
                        if (gradient_mode == MODE_WEIGHT_GRADIENT) begin
                            result_shift_r <= weight_gradient_shift;
                            rounding_bias_r <= round_bias(weight_gradient_shift);
                        end else begin
                            result_shift_r <= input_gradient_shift;
                            rounding_bias_r <= round_bias(input_gradient_shift);
                        end
                        mode_r <= gradient_mode;
                        compute_mode_r <= requested_accumulator_mode;
                        suppress_output_r <= suppress_output;
                        accumulate_result_r <= accumulate_result;
                        start_configuration_invalid_r
                            <= start_configuration_invalid;
                        requested_accumulator_mode_r
                            <= requested_accumulator_mode;
                        requested_accumulator_rows_r
                            <= requested_accumulator_rows;
                        requested_accumulator_columns_r
                            <= requested_accumulator_columns;
                        input_gradient_lane_r <= {RESULT_LANES{
                            requested_accumulator_mode
                                == MODE_INPUT_GRADIENT}};
                        if (requested_accumulator_mode
                                == MODE_INPUT_GRADIENT) begin
                            compute_result_row_count_r <= active_rows;
                            compute_result_column_count_r <= active_inputs;
                            compute_reduction_count_r <= active_outputs;
                        end else begin
                            compute_result_row_count_r <= active_inputs;
                            compute_result_column_count_r <= active_outputs;
                            compute_reduction_count_r <= active_rows;
                        end
                        load_row_r <= 4'd0;
                        load_half_r <= 1'b0;
                        result_row_r <= 4'd0;
                        column_group_r <= 1'b0;
                        reduction_group_r <= 1'b0;
                        output_word_r <= 2'd0;
                        config_error <= 1'b0;
                        for (int row = 0; row < LANES; row++) begin
                            result_row_buffer_r[row] <= 32'sd0;
                        end
                        state <= ST_CONFIGURE;
                    end
                end

                ST_CONFIGURE: begin
                    config_error <= start_configuration_invalid_r;
                    if (start_configuration_invalid_r) begin
                        state <= ST_DONE;
                    end else begin
                        accumulator_valid_r <= 1'b1;
                        accumulator_mode_r <= requested_accumulator_mode_r;
                        accumulator_rows_r <= requested_accumulator_rows_r;
                        accumulator_columns_r
                            <= requested_accumulator_columns_r;
                        if (mode_r == MODE_INPUT_GRADIENT) begin
                            load_final_r <= inputs_r == 4'd1;
                            state <= ST_LOAD_W;
                        end else begin
                            load_final_r <= rows_r == 4'd1;
                            state <= ST_LOAD_X;
                        end
                    end
                end

                ST_LOAD_X: if (input_stage_valid_r) begin
                    if (input_stage_last_r) config_error <= 1'b1;
                    if (!load_half_r) load_half_r <= 1'b1;
                    else if (load_final_r) begin
                        load_row_r <= 4'd0;
                        load_half_r <= 1'b0;
                        if (mode_r == MODE_WEIGHT_GRADIENT) begin
                            load_final_r <= rows_r == 4'd1;
                            state <= ST_LOAD_DY;
                        end else begin
                            load_final_r <= inputs_r == 4'd1;
                            state <= ST_LOAD_W;
                        end
                    end else begin
                        load_row_r <= load_row_r + 1'b1;
                        load_half_r <= 1'b0;
                        load_final_r <= load_row_r == rows_r - 4'd2;
                    end
                end

                ST_LOAD_W: if (input_stage_valid_r) begin
                    if (input_stage_last_r) config_error <= 1'b1;
                    if (!load_half_r) load_half_r <= 1'b1;
                    else if (load_final_r) begin
                        load_row_r <= 4'd0;
                        load_half_r <= 1'b0;
                        load_final_r <= rows_r == 4'd1;
                        state <= ST_LOAD_DY;
                    end else begin
                        load_row_r <= load_row_r + 1'b1;
                        load_half_r <= 1'b0;
                        load_final_r <= load_row_r == inputs_r - 4'd2;
                    end
                end

                ST_LOAD_DY: if (input_stage_valid_r) begin
                    if (input_stage_last_r != (load_half_r && load_final_r)) begin
                        config_error <= 1'b1;
                    end
                    if (!load_half_r) load_half_r <= 1'b1;
                    else if (load_final_r) begin
                        result_row_r <= 4'd0;
                        column_group_r <= 1'b0;
                        reduction_group_r <= 1'b0;
                        state <= ST_LOAD_COMMIT_DRAIN;
                    end else begin
                        load_row_r <= load_row_r + 1'b1;
                        load_half_r <= 1'b0;
                        load_final_r <= load_row_r == rows_r - 4'd2;
                    end
                end

                // The final dY payload has reached the commit registers. One
                // dedicated cycle lets its row-local write land before the DSP
                // operand-selection logic is allowed to read that bank.
                ST_LOAD_COMMIT_DRAIN: state <= ST_MULTIPLY;

                ST_MULTIPLY: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        for (int reduction_lane = 0;
                                reduction_lane < REDUCTION_LANES;
                                reduction_lane++) begin
                            multiplier_left_r[result_lane][reduction_lane]
                                <= multiplier_left[result_lane][reduction_lane];
                            multiplier_right_r[result_lane][reduction_lane]
                                <= multiplier_right[result_lane][reduction_lane];
                        end
                    end
                    state <= ST_REGISTER_OPERANDS;
                end

                ST_REGISTER_OPERANDS: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        for (int reduction_lane = 0;
                                reduction_lane < REDUCTION_LANES;
                                reduction_lane++) begin
                            multiplier_left_dsp_r[result_lane][reduction_lane]
                                <= multiplier_left_r[result_lane][reduction_lane];
                            multiplier_right_dsp_r[result_lane][reduction_lane]
                                <= multiplier_right_r[result_lane][reduction_lane];
                        end
                    end
                    state <= ST_PRODUCT;
                end

                ST_PRODUCT: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        for (int reduction_lane = 0;
                                reduction_lane < REDUCTION_LANES;
                                reduction_lane++) begin
                            product_r[result_lane][reduction_lane]
                                <= multiplier_product[result_lane][reduction_lane];
                        end
                    end
                    state <= ST_PAIR;
                end

                ST_PAIR: begin
                    state <= ST_REDUCTION_SUM;
                end

                ST_REDUCTION_SUM: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        reduction_sum_r[result_lane] <=
                            {pair_sum_r[result_lane][0][32],
                             pair_sum_r[result_lane][0]}
                            + {pair_sum_r[result_lane][1][32],
                               pair_sum_r[result_lane][1]};
                    end
                    state <= ST_REDUCE;
                end

                ST_REDUCE: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (!reduction_group_r) begin
                            row_partial_r[result_lane]
                                <= {reduction_sum_r[result_lane][33],
                                    reduction_sum_r[result_lane]};
                        end else begin
                            row_partial_r[result_lane] <=
                                row_partial_r[result_lane]
                                + {reduction_sum_r[result_lane][33],
                                   reduction_sum_r[result_lane]};
                        end
                    end
                    state <= ST_COMMIT;
                end

                ST_COMMIT: begin
                    if (!reduction_group_r && reduction_count > 4'd4) begin
                        reduction_group_r <= 1'b1;
                        state <= ST_MULTIPLY;
                    end else begin
                        for (int result_lane = 0; result_lane < RESULT_LANES;
                                result_lane++) begin
                            if (result_column_index[result_lane]
                                    < result_column_count) begin
                                if (accumulate_result_r) begin
                                    selected_accumulator_r[result_lane]
                                        <= accumulator_bank_read_data
                                            [result_lane];
                                end else begin
                                    selected_accumulator_r[result_lane]
                                        <= 47'sd0;
                                end
                                commit_lane_valid_r[result_lane] <= 1'b1;
                            end else begin
                                selected_accumulator_r[result_lane] <= 47'sd0;
                                commit_lane_valid_r[result_lane] <= 1'b0;
                            end
                        end
                        state <= ST_ADD_ACCUMULATOR;
                    end
                end

                ST_ADD_ACCUMULATOR: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        accumulated_wide_r[result_lane] <=
                            {selected_accumulator_r[result_lane][46],
                             selected_accumulator_r[result_lane]}
                            + {{13{row_partial_r[result_lane][34]}},
                               row_partial_r[result_lane]};
                    end
                    state <= ST_STORE_ACCUMULATOR;
                end

                ST_STORE_ACCUMULATOR: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (commit_lane_valid_r[result_lane]) begin
                            rounding_value_r[result_lane]
                                <= saturate_s47(accumulated_wide_r[result_lane]);
                            rounding_negative_r[result_lane]
                                <= accumulated_wide_r[result_lane][47];
                            if (accumulated_wide_r[result_lane][47]
                                    != accumulated_wide_r[result_lane][46]) begin
                                config_error <= 1'b1;
                            end
                        end else begin
                            rounding_value_r[result_lane] <= 47'd0;
                            rounding_negative_r[result_lane] <= 1'b0;
                        end
                    end
                    reduction_group_r <= 1'b0;
                    state <= ST_ROUND_MAGNITUDE;
                end

                ST_ROUND_MAGNITUDE: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (rounding_negative_r[result_lane]) begin
                            {rounding_carry_r[result_lane],
                             rounding_value_r[result_lane][23:0]} <=
                                {1'b0, ~rounding_value_r[result_lane][23:0]}
                                + 25'd1;
                        end else rounding_carry_r[result_lane] <= 1'b0;
                    end
                    state <= ST_ROUND_MAGNITUDE_HIGH;
                end

                ST_ROUND_MAGNITUDE_HIGH: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (rounding_negative_r[result_lane]) begin
                            rounding_value_r[result_lane][46:24] <=
                                (~rounding_value_r[result_lane][46:24])
                                + rounding_carry_r[result_lane];
                        end
                    end
                    state <= ST_ROUND_BIAS;
                end

                ST_ROUND_BIAS: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        {rounding_carry_r[result_lane],
                         rounding_value_r[result_lane][23:0]} <=
                            {1'b0, rounding_value_r[result_lane][23:0]}
                            + {1'b0, rounding_bias_r[23:0]};
                    end
                    state <= ST_ROUND_BIAS_HIGH;
                end

                ST_ROUND_BIAS_HIGH: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        rounding_value_r[result_lane][46:24] <=
                            rounding_value_r[result_lane][46:24]
                            + rounding_bias_r[46:24]
                            + rounding_carry_r[result_lane];
                    end
                    state <= ST_ROUND_SHIFT;
                end

                ST_ROUND_SHIFT: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        rounding_value_r[result_lane]
                            <= rounding_value_r[result_lane] >> result_shift_r;
                    end
                    state <= ST_ROUND_SIGN;
                end

                ST_ROUND_SIGN: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (rounding_negative_r[result_lane]) begin
                            {rounding_carry_r[result_lane],
                             rounding_value_r[result_lane][23:0]} <=
                                {1'b0, ~rounding_value_r[result_lane][23:0]}
                                + 25'd1;
                        end else rounding_carry_r[result_lane] <= 1'b0;
                    end
                    state <= ST_ROUND_SIGN_HIGH;
                end

                ST_ROUND_SIGN_HIGH: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (rounding_negative_r[result_lane]) begin
                            rounding_value_r[result_lane][46:24] <=
                                (~rounding_value_r[result_lane][46:24])
                                + rounding_carry_r[result_lane];
                        end
                    end
                    state <= ST_SATURATE;
                end

                ST_SATURATE: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (result_column_index[result_lane]
                                < result_column_count) begin
                            if (input_gradient_lane_r[result_lane]) begin
                                saturated_result_r[result_lane]
                                    <= saturate_s32($signed(
                                        rounding_value_r[result_lane]));
                            end else begin
                                saturated_result_r[result_lane]
                                    <= saturate_s24_extended($signed(
                                        rounding_value_r[result_lane]));
                            end
                        end else begin
                            saturated_result_r[result_lane] <= 32'sd0;
                        end
                    end
                    state <= ST_STORE_RESULT;
                end

                ST_STORE_RESULT: begin
                    for (int result_lane = 0; result_lane < RESULT_LANES;
                            result_lane++) begin
                        if (column_group_r) begin
                            result_row_buffer_r[result_lane + RESULT_LANES]
                                <= saturated_result_r[result_lane];
                        end else begin
                            result_row_buffer_r[result_lane]
                                <= saturated_result_r[result_lane];
                        end
                    end

                    if (!column_group_r) begin
                        column_group_r <= 1'b1;
                        state <= ST_MULTIPLY;
                    end else if (suppress_output_r) begin
                        column_group_r <= 1'b0;
                        if (result_row_r == result_row_count - 1'b1) begin
                            if (compute_mode_r == MODE_INPUT_GRADIENT
                                    && mode_r == MODE_BOTH) begin
                                state <= ST_SWITCH_DW;
                            end else begin
                                completed_tiles <= completed_tiles + 1'b1;
                                state <= ST_DONE;
                            end
                        end else begin
                            result_row_r <= result_row_r + 1'b1;
                            state <= ST_MULTIPLY;
                        end
                    end else begin
                        output_word_r <= 2'd0;
                        if (compute_mode_r == MODE_INPUT_GRADIENT) begin
                            state <= ST_OUTPUT_DX;
                        end else begin
                            state <= ST_OUTPUT_DW;
                        end
                    end
                end

                ST_OUTPUT_DX: if (m_axis_tready) begin
                    if (output_word_r == 3) begin
                        output_word_r <= 2'd0;
                        column_group_r <= 1'b0;
                        reduction_group_r <= 1'b0;
                        if (result_row_r == rows_r - 1'b1) begin
                            if (mode_r == MODE_BOTH) state <= ST_SWITCH_DW;
                            else begin
                                completed_tiles <= completed_tiles + 1'b1;
                                accumulator_valid_r <= 1'b0;
                                state <= ST_DONE;
                            end
                        end else begin
                            result_row_r <= result_row_r + 1'b1;
                            state <= ST_MULTIPLY;
                        end
                    end else output_word_r <= output_word_r + 1'b1;
                end

                ST_OUTPUT_DW: if (m_axis_tready) begin
                    if (output_word_r == 2) begin
                        output_word_r <= 2'd0;
                        column_group_r <= 1'b0;
                        reduction_group_r <= 1'b0;
                        if (result_row_r == inputs_r - 1'b1) begin
                            completed_tiles <= completed_tiles + 1'b1;
                            accumulator_valid_r <= 1'b0;
                            state <= ST_DONE;
                        end else begin
                            result_row_r <= result_row_r + 1'b1;
                            state <= ST_MULTIPLY;
                        end
                    end else output_word_r <= output_word_r + 1'b1;
                end

                ST_SWITCH_DW: begin
                    for (int row = 0; row < LANES; row++) begin
                        result_row_buffer_r[row] <= 32'sd0;
                    end
                    compute_mode_r <= MODE_WEIGHT_GRADIENT;
                    input_gradient_lane_r <= {RESULT_LANES{1'b0}};
                    compute_result_row_count_r <= inputs_r;
                    compute_result_column_count_r <= outputs_r;
                    compute_reduction_count_r <= rows_r;
                    result_shift_r <= dw_shift_r;
                    rounding_bias_r <= round_bias(dw_shift_r);
                    accumulator_mode_r <= MODE_WEIGHT_GRADIENT;
                    accumulator_rows_r <= inputs_r;
                    accumulator_columns_r <= outputs_r;
                    result_row_r <= 4'd0;
                    column_group_r <= 1'b0;
                    reduction_group_r <= 1'b0;
                    output_word_r <= 2'd0;
                    state <= ST_MULTIPLY;
                end

                ST_DONE: state <= ST_IDLE;
                default: state <= ST_IDLE;
            endcase

            if (clear_counters) begin
                completed_tiles <= 32'd0;
                active_cycles <= 32'd0;
                accumulator_valid_r <= 1'b0;
            end
        end
    end

    initial begin
        if (LANES != 8) begin
            $fatal(1, "neuraedge_linear_backward8 currently requires LANES=8");
        end
    end
endmodule

`default_nettype wire
