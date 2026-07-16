`timescale 1ns / 1ps
`default_nettype none

module neuraedge_buffered_tensor_core #(
    parameter int MAX_OUTER_ISSUES = 2048,
    parameter int MAX_DOT8_ISSUES = 256,
    parameter int COLUMN_LANES = 16
) (
    input  wire clk,
    input  wire rst_n,

    input  wire command_valid,
    output logic command_ready,
    input  wire [41:0] command_data,
    output logic command_done,
    output logic status_error,
    output logic status_busy,

    input  wire s_axis_tvalid,
    output logic s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire s_axis_tlast,

    output logic m_axis_tvalid,
    input  wire m_axis_tready,
    output logic [63:0] m_axis_tdata,
    output logic m_axis_tlast,

    output logic [11:0] status_loaded_activations,
    output logic [11:0] status_loaded_weights,
    output logic [11:0] status_active_issues,
    output logic [31:0] status_input_beats,
    output logic [31:0] status_output_beats,
    output logic [31:0] status_compute_cycles,
    output logic [31:0] status_skipped_issues,
    output logic [63:0] status_dense_macs,
    output logic [63:0] status_spike_adds
);
    import neuraedge_command_pkg::*;

    localparam int WEIGHT_BANKS = 8;

    typedef enum logic [3:0] {
        ST_IDLE = 4'd0,
        ST_LOAD_ACTIVATIONS = 4'd1,
        ST_LOAD_WEIGHTS_OUTER = 4'd2,
        ST_LOAD_WEIGHTS_DOT8 = 4'd3,
        ST_COMPUTE_START = 4'd4,
        ST_COMPUTE_RUN = 4'd5,
        ST_COMPUTE_OUTPUT = 4'd6,
        ST_LOAD_ACTIVATION_DRAIN = 4'd7,
        ST_DECODE_COMMAND = 4'd8,
        ST_COMPUTE_CONFIG = 4'd9,
        ST_LOAD_CONFIG = 4'd10,
        ST_COMPUTE_VALIDATE = 4'd11,
        ST_COMPUTE_VALIDATED = 4'd12,
        ST_PREDECODE_COMMAND = 4'd13,
        ST_COMPUTE_ACCOUNT = 4'd14,
        ST_COMPUTE_DRAIN = 4'd15
    } state_t;

    state_t state;
    logic [41:0] captured_command_r;
    logic [3:0] decoded_opcode_r;
    logic [11:0] decoded_issue_count_r;
    logic decoded_outer_issue_error_r;
    logic decoded_dot8_issue_error_r;
    (* dont_touch = "yes" *) logic command_done_pending_r;
    (* dont_touch = "yes" *) logic error_event_r;
    (* dont_touch = "yes" *) logic clear_error_r;
    logic [3:0] pending_load_opcode_r;
    logic [11:0] pending_load_issue_count_r;
    logic [7:0] pending_load_activation_zp_r;

    // Eight 256-deep, 128-bit banks are the physical form of a 32 KiB weight
    // tile. In
    // outer-product mode one bank supplies each consecutive K position. In
    // Dot8 mode reads all eight banks together, exposing 8x16 weight bytes.
    logic activation_write_enable;
    logic [10:0] activation_write_address;
    logic [10:0] activation_read_address;
    wire [63:0] activation_read_data;
    logic [WEIGHT_BANKS-1:0] weight_write_enable;
    logic [7:0] weight_write_address;
    logic [1:0] weight_write_segment;
    logic [WEIGHT_BANKS-1:0] weight_commit_enable_r;
    logic [7:0] weight_commit_address_r;
    logic [1:0] weight_commit_segment_r;
    logic [63:0] weight_commit_data_r;
    logic [7:0] weight_read_address;
    wire [COLUMN_LANES*8-1:0] weight_read_data [0:WEIGHT_BANKS-1];

    logic [11:0] load_requested_issues;
    logic [14:0] load_beats_remaining_r;
    logic load_finish_pending_r;
    logic [3:0] load_segment_index_r;
    logic [11:0] load_item_index;
    logic [7:0] load_activation_zero_point;
    logic weight_format_dot8;
    logic activation_classify_valid_r;
    logic activation_classify_nonzero_r;
    logic [10:0] activation_classify_index_r;
    logic activation_load_success_r;
    wire active_index_write_enable;
    wire [10:0] active_index_read_data;
    wire active_index_read_enable;
    wire [10:0] active_mapped_index;

    (* dont_touch = "yes" *) logic [1:0] compute_mode_r;
    (* dont_touch = "yes" *) logic [3:0] compute_rows_r;
    (* dont_touch = "yes" *) logic [5:0] compute_columns_r;
    logic [11:0] compute_requested_issues;
    logic compute_validation_error_r;
    (* dont_touch = "yes" *) logic [7:0] compute_activation_zp_r;
    (* dont_touch = "yes" *) logic [7:0] compute_weight_zp_r;
    logic [11:0] schedule_pointer;
    logic [11:0] schedule_remaining_r;
    logic schedule_active_r;
    logic schedule_last_r;
    logic compute_empty_r;
    logic compute_read_request;
    logic index_request_valid_r;
    logic index_request_last_r;
    logic [10:0] index_request_address_r;
    logic index_valid_r;
    logic index_last_r;
    logic map_valid_r;
    logic map_last_r;
    logic [10:0] mapped_index_r;
    logic read_valid_r;
    logic [2:0] read_bank_select_r;
    logic read_last_r;
    logic memory_valid_r;
    logic [2:0] memory_bank_select_r;
    logic memory_last_r;
    logic capture_valid_r;
    logic capture_last_r;
    logic [63:0] capture_activation_r;
    logic [COLUMN_LANES*8-1:0] capture_outer_weights_r;
    logic [COLUMN_LANES*8*8-1:0] capture_dot_weights_r;
    logic [1:0] issue_fifo_count_r;
    logic issue_fifo_write_pointer_r;
    logic issue_fifo_read_pointer_r;
    logic issue_fifo_last_r [0:1];
    logic [63:0] issue_fifo_activation_r [0:1];
    logic [COLUMN_LANES*8-1:0] issue_fifo_outer_weights_r [0:1];
    logic [COLUMN_LANES*8*8-1:0] issue_fifo_dot_weights_r [0:1];
    (* dont_touch = "yes" *) logic tile_last_issue_accepted_r;
    logic issue_fifo_available;
    logic capture_stage_available;
    logic memory_stage_available;
    logic read_stage_available;
    logic map_stage_available;
    logic index_stage_available;
    logic index_request_stage_available;

    logic tile_start;
    logic tile_config_error;
    logic tile_issue_valid;
    wire tile_issue_ready;
    logic [63:0] tile_activation_vector;
    logic [COLUMN_LANES*8-1:0] tile_outer_weight_vector;
    logic [COLUMN_LANES*8*8-1:0] tile_dot_weight_vector;
    logic tile_issue_last;
    wire tile_result_valid;
    logic tile_result_ready;
    wire signed [47:0] tile_result_value;
    wire [3:0] tile_result_row;
    wire [5:0] tile_result_column;
    wire tile_result_last;
    // This two-entry skid FIFO breaks the external READY path before it reaches
    // the tile's 128-entry result store and replicated output-state controls.
    // A registered tile READY can remain asserted for one final transfer while
    // downstream backpressure arrives; the second entry is the space reserved
    // for that transfer, so no result is dropped and sustained throughput stays
    // at one result per clock whenever the parent accepts one per clock.
    logic tile_result_ready_r;
    logic [1:0] tensor_output_fifo_count_r;
    logic tensor_output_fifo_write_pointer_r;
    logic tensor_output_fifo_read_pointer_r;
    logic [63:0] tensor_output_fifo_data_r [0:1];
    logic tensor_output_fifo_last_r [0:1];
    wire [63:0] tile_debug_dense_macs;
    wire [63:0] tile_debug_spike_adds;
    logic [23:0] pending_dense_macs_low_r;
    logic pending_dense_macs_carry_r;
    logic [23:0] pending_spike_adds_low_r;
    logic pending_spike_adds_carry_r;
    wire issue_fifo_push = capture_valid_r && issue_fifo_available;
    wire issue_fifo_pop = issue_fifo_count_r != 0 && tile_issue_ready;
    wire tensor_output_fifo_push = state == ST_COMPUTE_OUTPUT
        && tile_result_valid && tile_result_ready_r;
    wire tensor_output_fifo_pop = tensor_output_fifo_count_r != 0
        && m_axis_tready;
    wire tensor_output_fifo_head_last = tensor_output_fifo_last_r[
        tensor_output_fifo_read_pointer_r];
    logic [1:0] tensor_output_fifo_count_next;

    function automatic logic all_bytes_equal64(
        input logic [63:0] values,
        input logic [7:0] reference
    );
        logic equal_all;
        begin
            equal_all = 1'b1;
            for (int lane = 0; lane < 8; lane++) begin
                equal_all = equal_all && (values[lane*8 +: 8] == reference);
            end
            all_bytes_equal64 = equal_all;
        end
    endfunction

    function automatic logic [63:0] repeat_byte(input logic [7:0] value);
        begin
            repeat_byte = {8{value}};
        end
    endfunction

    // Tile activity is at most 19 bits per command.  The low 24-bit addition
    // and its carry are registered on the final output beat; the following
    // accounting state updates the independent high 40 bits.
    function automatic logic [24:0] add_u19_to_u24(
        input logic [23:0] value,
        input logic [18:0] increment
    );
        begin
            add_u19_to_u24 = {1'b0, value} + {6'd0, increment};
        end
    endfunction

    assign tile_start = (state == ST_COMPUTE_START);
    assign active_index_write_enable = activation_classify_valid_r
        && activation_classify_nonzero_r;
    // Each accepted request generates exactly one synchronous BRAM response.
    // The following local register stage holds that response during downstream
    // stalls, so an unaccepted address can never overwrite accepted metadata.
    assign active_index_read_enable = index_request_valid_r && !compute_empty_r;
    assign active_mapped_index = compute_empty_r
        ? 11'd0 : active_index_read_data;

    neuraedge_active_index_buffer #(
        .DEPTH(MAX_OUTER_ISSUES),
        .ADDRESS_BITS(11)
    ) u_active_index_buffer (
        .clk(clk),
        .write_enable(active_index_write_enable),
        .write_address(status_active_issues[10:0]),
        .write_data(activation_classify_index_r),
        .read_enable(active_index_read_enable),
        .read_address(index_request_address_r),
        .read_data(active_index_read_data)
    );

    neuraedge_activation_buffer u_activation_buffer (
        .clk(clk),
        .write_enable(activation_write_enable),
        .write_address(activation_write_address),
        .write_data(s_axis_tdata),
        .read_address(activation_read_address),
        .read_data(activation_read_data)
    );

    for (genvar bank = 0; bank < WEIGHT_BANKS; bank++) begin : g_weight_bank
        neuraedge_weight_bank u_weight_bank (
            .clk(clk),
            .write_enable(weight_commit_enable_r[bank]),
            .write_address(weight_commit_address_r),
            .write_segment(weight_commit_segment_r),
            .write_data(weight_commit_data_r),
            .read_address(weight_read_address),
            .read_data(weight_read_data[bank])
        );
    end

    neuraedge_tensor_tile #(.COLUMN_LANES(COLUMN_LANES)) u_tensor_tile (
        .clk(clk),
        .rst_n(rst_n),
        .start(tile_start),
        .compute_mode(compute_mode_r),
        .active_rows(compute_rows_r),
        .active_columns(compute_columns_r),
        .activation_zero_point(compute_activation_zp_r),
        .weight_zero_point(compute_weight_zp_r),
        .config_error(tile_config_error),
        .issue_valid(tile_issue_valid),
        .issue_ready(tile_issue_ready),
        .activation_vector(tile_activation_vector),
        .outer_weight_vector(tile_outer_weight_vector),
        .dot_weight_vector(tile_dot_weight_vector),
        .issue_last(tile_issue_last),
        .busy(),
        .result_valid(tile_result_valid),
        .result_ready(tile_result_ready),
        .result_value(tile_result_value),
        .result_row(tile_result_row),
        .result_column(tile_result_column),
        .result_last(tile_result_last),
        .debug_accepted_issues(),
        .debug_skipped_issues(),
        .debug_retired_issues(),
        .debug_dense_macs(tile_debug_dense_macs),
        .debug_spike_adds(tile_debug_spike_adds)
    );

`ifndef SYNTHESIS
    // The tile's mathematical per-command maximum is 262,144, so its debug
    // counters must remain 19-bit quantities.  Keep the fast accumulator's
    // width assumption checked in every RTL simulation.
    always_ff @(posedge clk) begin
        if (rst_n && tile_result_valid && tile_result_ready && tile_result_last
                && ((|tile_debug_dense_macs[63:19])
                    || (|tile_debug_spike_adds[63:19]))) begin
            $fatal(1, "tile activity counter exceeded its 19-bit contract");
        end
    end
`endif

    always_comb begin
        command_ready = (state == ST_IDLE);
        status_busy = (state != ST_IDLE);
        s_axis_tready = !load_finish_pending_r
            && (state == ST_LOAD_ACTIVATIONS
                || state == ST_LOAD_WEIGHTS_OUTER
                || state == ST_LOAD_WEIGHTS_DOT8);

        m_axis_tvalid = tensor_output_fifo_count_r != 0;
        m_axis_tdata = tensor_output_fifo_data_r[
            tensor_output_fifo_read_pointer_r];
        m_axis_tlast = tensor_output_fifo_head_last;
        tile_result_ready = state == ST_COMPUTE_OUTPUT
            && tile_result_ready_r;

        activation_write_enable = state == ST_LOAD_ACTIVATIONS
            && s_axis_tvalid && s_axis_tready;
        activation_write_address = load_item_index[10:0];
        weight_write_enable = '0;
        weight_write_address = 8'd0;
        weight_write_segment = {1'b0, load_segment_index_r[0]};
        if (state == ST_LOAD_WEIGHTS_OUTER && s_axis_tvalid && s_axis_tready) begin
            weight_write_enable[load_item_index[2:0]] = 1'b1;
            weight_write_address = load_item_index[10:3];
        end else if (state == ST_LOAD_WEIGHTS_DOT8 && s_axis_tvalid && s_axis_tready) begin
            weight_write_enable[load_segment_index_r[3:1]] = 1'b1;
            weight_write_address = load_item_index[7:0];
        end

        // A registered two-entry queue terminates tile READY before it can
        // propagate through every BRAM-response stage. Deliberately refusing
        // a same-cycle push while full costs at most one recovery cycle and
        // keeps the queue credit independent of downstream combinational logic.
        issue_fifo_available = issue_fifo_count_r != 2;
        capture_stage_available = !capture_valid_r || issue_fifo_available;
        memory_stage_available = !memory_valid_r || capture_stage_available;
        read_stage_available = !read_valid_r || memory_stage_available;
        map_stage_available = !map_valid_r || read_stage_available;
        index_stage_available = !index_valid_r || map_stage_available;
        index_request_stage_available = !index_request_valid_r
            || index_stage_available;
        compute_read_request = state == ST_COMPUTE_RUN
            && schedule_active_r
            && index_request_stage_available;
        activation_read_address = mapped_index_r;
        weight_read_address = compute_mode_r == MODE_DENSE_DOT8
            ? mapped_index_r[7:0] : mapped_index_r[10:3];

        tile_issue_valid = issue_fifo_count_r != 0;
        tile_issue_last = issue_fifo_last_r[issue_fifo_read_pointer_r];
        tile_activation_vector =
            issue_fifo_activation_r[issue_fifo_read_pointer_r];
        tile_outer_weight_vector =
            issue_fifo_outer_weights_r[issue_fifo_read_pointer_r];
        tile_dot_weight_vector =
            issue_fifo_dot_weights_r[issue_fifo_read_pointer_r];

        tensor_output_fifo_count_next = tensor_output_fifo_count_r;
        unique case ({tensor_output_fifo_push, tensor_output_fifo_pop})
            2'b10: tensor_output_fifo_count_next
                = tensor_output_fifo_count_r + 1'b1;
            2'b01: tensor_output_fifo_count_next
                = tensor_output_fifo_count_r - 1'b1;
            default: begin end
        endcase
    end

    // The FIFO owns the timing boundary independently of the command FSM.  Its
    // READY register is derived from post-transfer occupancy, which permits a
    // simultaneous dequeue/enqueue every clock and deasserts early enough to
    // reserve the second slot when the downstream interface stops.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tile_result_ready_r <= 1'b0;
            tensor_output_fifo_count_r <= 2'd0;
            tensor_output_fifo_write_pointer_r <= 1'b0;
            tensor_output_fifo_read_pointer_r <= 1'b0;
            for (int output_entry = 0; output_entry < 2; output_entry++) begin
                tensor_output_fifo_data_r[output_entry] <= 64'd0;
                tensor_output_fifo_last_r[output_entry] <= 1'b0;
            end
        end else begin
            if (state == ST_COMPUTE_START) begin
                tensor_output_fifo_count_r <= 2'd0;
                tensor_output_fifo_write_pointer_r <= 1'b0;
                tensor_output_fifo_read_pointer_r <= 1'b0;
            end else begin
                tensor_output_fifo_count_r <= tensor_output_fifo_count_next;
                if (tensor_output_fifo_push) begin
                    tensor_output_fifo_data_r[tensor_output_fifo_write_pointer_r]
                        <= {5'd0, tile_result_last, tile_result_column,
                            tile_result_row, tile_result_value};
                    tensor_output_fifo_last_r[tensor_output_fifo_write_pointer_r]
                        <= tile_result_last;
                    tensor_output_fifo_write_pointer_r
                        <= !tensor_output_fifo_write_pointer_r;
                end
                if (tensor_output_fifo_pop) begin
                    tensor_output_fifo_read_pointer_r
                        <= !tensor_output_fifo_read_pointer_r;
                end
            end

            tile_result_ready_r <= (state == ST_COMPUTE_OUTPUT
                    && tensor_output_fifo_count_next != 2)
                || (state == ST_COMPUTE_RUN && tile_last_issue_accepted_r);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            captured_command_r <= 42'd0;
            decoded_opcode_r <= 4'd0;
            decoded_issue_count_r <= 12'd0;
            decoded_outer_issue_error_r <= 1'b0;
            decoded_dot8_issue_error_r <= 1'b0;
            command_done_pending_r <= 1'b0;
            error_event_r <= 1'b0;
            clear_error_r <= 1'b0;
            if (state == ST_COMPUTE_RUN && tile_config_error) begin
                error_event_r <= 1'b1;
            end
            pending_load_opcode_r <= 4'd0;
            pending_load_issue_count_r <= 12'd0;
            pending_load_activation_zp_r <= 8'd0;
            command_done <= 1'b0;
            status_error <= 1'b0;
            status_loaded_activations <= 12'd0;
            status_loaded_weights <= 12'd0;
            status_active_issues <= 12'd0;
            status_input_beats <= 32'd0;
            status_output_beats <= 32'd0;
            status_compute_cycles <= 32'd0;
            status_skipped_issues <= 32'd0;
            status_dense_macs <= 64'd0;
            status_spike_adds <= 64'd0;
            pending_dense_macs_low_r <= 24'd0;
            pending_dense_macs_carry_r <= 1'b0;
            pending_spike_adds_low_r <= 24'd0;
            pending_spike_adds_carry_r <= 1'b0;
            load_requested_issues <= 12'd0;
            load_beats_remaining_r <= 15'd0;
            load_finish_pending_r <= 1'b0;
            load_segment_index_r <= 4'd0;
            load_item_index <= 12'd0;
            load_activation_zero_point <= 8'd0;
            weight_format_dot8 <= 1'b0;
            weight_commit_enable_r <= '0;
            weight_commit_address_r <= 8'd0;
            weight_commit_segment_r <= 2'd0;
            weight_commit_data_r <= 64'd0;
            activation_classify_valid_r <= 1'b0;
            activation_classify_nonzero_r <= 1'b0;
            activation_classify_index_r <= 11'd0;
            activation_load_success_r <= 1'b0;
            compute_mode_r <= MODE_DENSE_OUTER;
            compute_rows_r <= 4'd1;
            compute_columns_r <= 6'd1;
            compute_requested_issues <= 12'd0;
            compute_validation_error_r <= 1'b0;
            compute_activation_zp_r <= 8'd0;
            compute_weight_zp_r <= 8'd0;
            schedule_pointer <= 12'd0;
            schedule_remaining_r <= 12'd0;
            schedule_active_r <= 1'b0;
            schedule_last_r <= 1'b0;
            compute_empty_r <= 1'b0;
            index_request_valid_r <= 1'b0;
            index_request_last_r <= 1'b0;
            index_request_address_r <= 11'd0;
            index_valid_r <= 1'b0;
            index_last_r <= 1'b0;
            map_valid_r <= 1'b0;
            map_last_r <= 1'b0;
            mapped_index_r <= 11'd0;
            read_valid_r <= 1'b0;
            read_bank_select_r <= 3'd0;
            read_last_r <= 1'b0;
            memory_valid_r <= 1'b0;
            memory_bank_select_r <= 3'd0;
            memory_last_r <= 1'b0;
            capture_valid_r <= 1'b0;
            capture_last_r <= 1'b0;
            capture_activation_r <= 64'd0;
            capture_outer_weights_r <= '0;
            capture_dot_weights_r <= '0;
            issue_fifo_count_r <= 2'd0;
            issue_fifo_write_pointer_r <= 1'b0;
            issue_fifo_read_pointer_r <= 1'b0;
            tile_last_issue_accepted_r <= 1'b0;
            for (int fifo_entry = 0; fifo_entry < 2; fifo_entry++) begin
                issue_fifo_last_r[fifo_entry] <= 1'b0;
                issue_fifo_activation_r[fifo_entry] <= 64'd0;
                issue_fifo_outer_weights_r[fifo_entry] <= '0;
                issue_fifo_dot_weights_r[fifo_entry] <= '0;
            end
        end else begin
            // Register the accepted write request before it reaches sixteen
            // BRAM enable pins. This remains a one-beat-per-cycle pipeline:
            // the banks commit the prior request while this stage captures
            // the next one.
            weight_commit_enable_r <= weight_write_enable;
            if (|weight_write_enable) begin
                weight_commit_address_r <= weight_write_address;
                weight_commit_segment_r <= weight_write_segment;
                weight_commit_data_r <= s_axis_tdata;
            end
            command_done <= command_done_pending_r;
            command_done_pending_r <= 1'b0;
            if (clear_error_r) status_error <= 1'b0;
            else if (error_event_r) status_error <= 1'b1;
            error_event_r <= 1'b0;
            clear_error_r <= 1'b0;
            activation_classify_valid_r <= 1'b0;
            tile_last_issue_accepted_r <= tile_issue_valid
                && tile_issue_ready && tile_issue_last;

            if (issue_fifo_push) begin
                issue_fifo_last_r[issue_fifo_write_pointer_r] <= capture_last_r;
                issue_fifo_activation_r[issue_fifo_write_pointer_r]
                    <= capture_activation_r;
                issue_fifo_outer_weights_r[issue_fifo_write_pointer_r]
                    <= capture_outer_weights_r;
                issue_fifo_dot_weights_r[issue_fifo_write_pointer_r]
                    <= capture_dot_weights_r;
                issue_fifo_write_pointer_r <= !issue_fifo_write_pointer_r;
            end
            if (issue_fifo_pop) begin
                issue_fifo_read_pointer_r <= !issue_fifo_read_pointer_r;
            end
            unique case ({issue_fifo_push, issue_fifo_pop})
                2'b10: issue_fifo_count_r <= issue_fifo_count_r + 1'b1;
                2'b01: issue_fifo_count_r <= issue_fifo_count_r - 1'b1;
                default: begin end
            endcase

            if (activation_classify_valid_r && activation_classify_nonzero_r) begin
                status_active_issues <= status_active_issues + 1'b1;
            end

            if (state == ST_COMPUTE_START || state == ST_COMPUTE_RUN
                    || state == ST_COMPUTE_OUTPUT
                    || state == ST_COMPUTE_DRAIN) begin
                status_compute_cycles <= status_compute_cycles + 1'b1;
            end
            if (tensor_output_fifo_pop) begin
                status_output_beats <= status_output_beats + 1'b1;
            end

            unique case (state)
                ST_IDLE: begin
                    map_valid_r <= 1'b0;
                    read_valid_r <= 1'b0;
                    memory_valid_r <= 1'b0;
                    issue_fifo_count_r <= 2'd0;
                    issue_fifo_write_pointer_r <= 1'b0;
                    issue_fifo_read_pointer_r <= 1'b0;
                    if (command_valid) begin
                        captured_command_r <= command_data[41:0];
                        state <= ST_PREDECODE_COMMAND;
                    end
                end

                ST_PREDECODE_COMMAND: begin
                    decoded_opcode_r <= captured_command_r[3:0];
                    decoded_issue_count_r <= captured_command_r[25:14];
                    decoded_outer_issue_error_r <= captured_command_r[25:14] == 0
                        || captured_command_r[25:14] > 12'(MAX_OUTER_ISSUES);
                    decoded_dot8_issue_error_r <= captured_command_r[25:14] == 0
                        || captured_command_r[25:14] > 12'(MAX_DOT8_ISSUES);
                    state <= ST_DECODE_COMMAND;
                end

                ST_DECODE_COMMAND: begin
                    // Default back to idle for immediate commands and rejected
                    // requests. Valid streaming/compute commands override this.
                    state <= ST_IDLE;
                    begin
                        logic [3:0] opcode;
                        logic [11:0] issue_count;
                        opcode = decoded_opcode_r;
                        issue_count = decoded_issue_count_r;

                        unique case (opcode)
                            CMD_LOAD_ACTIVATIONS: begin
                                if (decoded_outer_issue_error_r) begin
                                    error_event_r <= 1'b1;
                                    status_loaded_activations <= 12'd0;
                                    status_active_issues <= 12'd0;
                                    command_done_pending_r <= 1'b1;
                                end else begin
                                    pending_load_opcode_r <= opcode;
                                    pending_load_issue_count_r <= issue_count;
                                    pending_load_activation_zp_r
                                        <= captured_command_r[33:26];
                                    state <= ST_LOAD_CONFIG;
                                end
                            end

                            CMD_LOAD_WEIGHTS_OUTER: begin
                                if (decoded_outer_issue_error_r) begin
                                    error_event_r <= 1'b1;
                                    status_loaded_weights <= 12'd0;
                                    command_done_pending_r <= 1'b1;
                                end else begin
                                    pending_load_opcode_r <= opcode;
                                    pending_load_issue_count_r <= issue_count;
                                    state <= ST_LOAD_CONFIG;
                                end
                            end

                            CMD_LOAD_WEIGHTS_DOT8: begin
                                if (decoded_dot8_issue_error_r) begin
                                    error_event_r <= 1'b1;
                                    status_loaded_weights <= 12'd0;
                                    command_done_pending_r <= 1'b1;
                                end else begin
                                    pending_load_opcode_r <= opcode;
                                    pending_load_issue_count_r <= issue_count;
                                    state <= ST_LOAD_CONFIG;
                                end
                            end

                            CMD_COMPUTE_DENSE_OUTER,
                            CMD_COMPUTE_DENSE_DOT8,
                            CMD_COMPUTE_BINARY,
                            CMD_COMPUTE_TERNARY: begin
                                unique case (opcode)
                                    CMD_COMPUTE_DENSE_OUTER:
                                        compute_mode_r <= MODE_DENSE_OUTER;
                                    CMD_COMPUTE_DENSE_DOT8:
                                        compute_mode_r <= MODE_DENSE_DOT8;
                                    CMD_COMPUTE_BINARY:
                                        compute_mode_r <= MODE_BINARY_WEIGHTED;
                                    default: compute_mode_r <= MODE_TERNARY_WEIGHTED;
                                endcase
                                compute_rows_r <= captured_command_r[7:4];
                                compute_columns_r <= captured_command_r[13:8];
                                compute_requested_issues <= issue_count;
                                compute_activation_zp_r <= captured_command_r[33:26];
                                compute_weight_zp_r <= captured_command_r[41:34];
                                state <= ST_COMPUTE_VALIDATE;
                            end

                            CMD_CLEAR_STATUS: begin
                                clear_error_r <= 1'b1;
                                status_input_beats <= 32'd0;
                                status_output_beats <= 32'd0;
                                status_compute_cycles <= 32'd0;
                                status_skipped_issues <= 32'd0;
                                status_dense_macs <= 64'd0;
                                status_spike_adds <= 64'd0;
                                command_done_pending_r <= 1'b1;
                            end

                            default: begin
                                error_event_r <= 1'b1;
                                command_done_pending_r <= 1'b1;
                            end
                        endcase
                    end
                end

                ST_LOAD_CONFIG: begin
                    load_requested_issues <= pending_load_issue_count_r;
                    load_segment_index_r <= 4'd0;
                    load_item_index <= 12'd0;
                    load_finish_pending_r <= 1'b0;
                    unique case (pending_load_opcode_r)
                        CMD_LOAD_ACTIVATIONS: begin
                            load_beats_remaining_r <=
                                {3'd0, pending_load_issue_count_r};
                            load_activation_zero_point <= pending_load_activation_zp_r;
                            activation_load_success_r <= 1'b0;
                            status_active_issues <= 12'd0;
                            state <= ST_LOAD_ACTIVATIONS;
                        end
                        CMD_LOAD_WEIGHTS_OUTER: begin
                            load_beats_remaining_r <=
                                {2'd0, pending_load_issue_count_r, 1'b0};
                            weight_format_dot8 <= 1'b0;
                            state <= ST_LOAD_WEIGHTS_OUTER;
                        end
                        default: begin
                            load_beats_remaining_r <=
                                {2'd0, pending_load_issue_count_r[8:0], 4'd0};
                            weight_format_dot8 <= 1'b1;
                            state <= ST_LOAD_WEIGHTS_DOT8;
                        end
                    endcase
                end

                ST_COMPUTE_CONFIG: begin
                    compute_empty_r <= (status_active_issues == 0);
                    schedule_remaining_r <= status_active_issues == 0
                        ? 12'd1 : status_active_issues;
                    schedule_active_r <= 1'b1;
                    schedule_last_r <= status_active_issues <= 1;
                    schedule_pointer <= 12'd0;
                    status_compute_cycles <= 32'd0;
                    status_output_beats <= 32'd0;
                    state <= ST_COMPUTE_START;
                end

                ST_COMPUTE_VALIDATE: begin
                    // End the multi-field legality predicate at one local
                    // register instead of placing it on every parameter mux.
                    compute_validation_error_r <= compute_requested_issues == 0
                        || compute_rows_r == 0
                        || compute_rows_r > 8
                        || compute_columns_r == 0
                        || compute_columns_r > 6'(COLUMN_LANES)
                        || (compute_mode_r == MODE_DENSE_DOT8
                            && compute_rows_r != 1)
                        || compute_requested_issues != status_loaded_activations
                        || compute_requested_issues != status_loaded_weights
                        || (compute_mode_r == MODE_DENSE_DOT8)
                            != weight_format_dot8;
                    state <= ST_COMPUTE_VALIDATED;
                end

                ST_COMPUTE_VALIDATED: begin
                    if (compute_validation_error_r) begin
                        error_event_r <= 1'b1;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end else begin
                        state <= ST_COMPUTE_CONFIG;
                    end
                end

                ST_LOAD_ACTIVATIONS: begin
                    if (load_finish_pending_r) begin
                        load_finish_pending_r <= 1'b0;
                        state <= ST_LOAD_ACTIVATION_DRAIN;
                    end else if (s_axis_tvalid && s_axis_tready) begin
                        logic beat_is_last;
                        logic byte_vector_is_zero;
                        beat_is_last = load_beats_remaining_r == 15'd1;
                        byte_vector_is_zero = all_bytes_equal64(
                            s_axis_tdata, load_activation_zero_point);
                        activation_classify_valid_r <= 1'b1;
                        activation_classify_nonzero_r <= !byte_vector_is_zero;
                        activation_classify_index_r <= load_item_index[10:0];
                        status_input_beats <= status_input_beats + 1'b1;
                        load_beats_remaining_r <= load_beats_remaining_r - 1'b1;
                        load_item_index <= load_item_index + 1'b1;
                        status_loaded_activations <= beat_is_last
                            ? load_requested_issues : 12'd0;

                        if (s_axis_tlast != beat_is_last) error_event_r <= 1'b1;
                        if (beat_is_last || s_axis_tlast) begin
                            activation_load_success_r <= beat_is_last;
                            load_finish_pending_r <= 1'b1;
                        end
                    end
                end

                ST_LOAD_ACTIVATION_DRAIN: begin
                    if (!activation_classify_valid_r) begin
                        if (!activation_load_success_r) status_active_issues <= 12'd0;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                ST_LOAD_WEIGHTS_OUTER: begin
                    if (load_finish_pending_r) begin
                        load_finish_pending_r <= 1'b0;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end else if (s_axis_tvalid && s_axis_tready) begin
                        logic [1:0] word_segment;
                        logic beat_is_last;
                        word_segment = {1'b0, load_segment_index_r[0]};
                        beat_is_last = load_beats_remaining_r == 15'd1;
                        status_input_beats <= status_input_beats + 1'b1;
                        load_beats_remaining_r <= load_beats_remaining_r - 1'b1;
                        status_loaded_weights <= beat_is_last
                            ? load_requested_issues : 12'd0;
                        if (word_segment == 2'd1) begin
                            load_item_index <= load_item_index + 1'b1;
                            load_segment_index_r <= 4'd0;
                        end else begin
                            load_segment_index_r <= 4'd1;
                        end

                        if (s_axis_tlast != beat_is_last) error_event_r <= 1'b1;
                        if (beat_is_last || s_axis_tlast) begin
                            load_finish_pending_r <= 1'b1;
                        end
                    end
                end

                ST_LOAD_WEIGHTS_DOT8: begin
                    if (load_finish_pending_r) begin
                        load_finish_pending_r <= 1'b0;
                        command_done_pending_r <= 1'b1;
                        state <= ST_IDLE;
                    end else if (s_axis_tvalid && s_axis_tready) begin
                        logic beat_is_last;
                        beat_is_last = load_beats_remaining_r == 15'd1;
                        status_input_beats <= status_input_beats + 1'b1;
                        load_beats_remaining_r <= load_beats_remaining_r - 1'b1;
                        status_loaded_weights <= beat_is_last
                            ? load_requested_issues : 12'd0;
                        if (load_segment_index_r == 4'd15) begin
                            load_item_index <= load_item_index + 1'b1;
                            load_segment_index_r <= 4'd0;
                        end else begin
                            load_segment_index_r <= load_segment_index_r + 1'b1;
                        end

                        if (s_axis_tlast != beat_is_last) error_event_r <= 1'b1;
                        if (beat_is_last || s_axis_tlast) begin
                            load_finish_pending_r <= 1'b1;
                        end
                    end
                end

                ST_COMPUTE_START: begin
                    schedule_pointer <= 12'd0;
                    index_request_valid_r <= 1'b0;
                    index_valid_r <= 1'b0;
                    map_valid_r <= 1'b0;
                    read_valid_r <= 1'b0;
                    memory_valid_r <= 1'b0;
                    capture_valid_r <= 1'b0;
                    issue_fifo_count_r <= 2'd0;
                    issue_fifo_write_pointer_r <= 1'b0;
                    issue_fifo_read_pointer_r <= 1'b0;
                    state <= ST_COMPUTE_RUN;
                end

                ST_COMPUTE_RUN: begin
                    if (capture_stage_available) begin
                        capture_valid_r <= memory_valid_r;
                        if (memory_valid_r) begin
                            capture_last_r <= memory_last_r;
                            if (compute_empty_r) begin
                                capture_activation_r <= (compute_mode_r == MODE_DENSE_OUTER
                                        || compute_mode_r == MODE_DENSE_DOT8)
                                    ? repeat_byte(compute_activation_zp_r) : 64'd0;
                                capture_outer_weights_r <= '0;
                                capture_dot_weights_r <= '0;
                            end else begin
                                capture_activation_r <= activation_read_data;
                                capture_outer_weights_r
                                    <= weight_read_data[memory_bank_select_r];
                                capture_dot_weights_r <= {
                                    weight_read_data[7], weight_read_data[6],
                                    weight_read_data[5], weight_read_data[4],
                                    weight_read_data[3], weight_read_data[2],
                                    weight_read_data[1], weight_read_data[0]
                                };
                            end
                        end
                    end

                    if (memory_stage_available) begin
                        memory_valid_r <= read_valid_r;
                        if (read_valid_r) begin
                            memory_bank_select_r <= read_bank_select_r;
                            memory_last_r <= read_last_r;
                        end
                    end

                    if (read_stage_available) begin
                        read_valid_r <= map_valid_r;
                        if (map_valid_r) begin
                            read_bank_select_r <= mapped_index_r[2:0];
                            read_last_r <= map_last_r;
                        end
                    end

                    if (map_stage_available) begin
                        map_valid_r <= index_valid_r;
                        if (index_valid_r) begin
                            mapped_index_r <= active_mapped_index;
                            map_last_r <= index_last_r;
                        end
                    end

                    if (index_stage_available) begin
                        index_valid_r <= index_request_valid_r;
                        if (index_request_valid_r) begin
                            index_last_r <= index_request_last_r;
                        end
                    end

                    if (index_request_stage_available) begin
                        index_request_valid_r <= compute_read_request;
                        if (compute_read_request) begin
                            index_request_address_r <= schedule_pointer[10:0];
                            index_request_last_r <= schedule_last_r;
                            schedule_last_r <= schedule_remaining_r == 12'd2;
                            schedule_remaining_r <= schedule_remaining_r - 1'b1;
                            if (schedule_remaining_r == 12'd1) begin
                                schedule_active_r <= 1'b0;
                            end
                            schedule_pointer <= schedule_pointer + 1'b1;
                        end
                    end

                    if (tile_last_issue_accepted_r) begin
                        state <= ST_COMPUTE_OUTPUT;
                    end
                end

                ST_COMPUTE_OUTPUT: begin
                    status_skipped_issues <= {20'd0, compute_requested_issues}
                        - {20'd0, status_active_issues};
                    if (tensor_output_fifo_push) begin
                        if (tile_result_last) begin
                            {pending_dense_macs_carry_r,
                             pending_dense_macs_low_r}
                                <= add_u19_to_u24(
                                    status_dense_macs[23:0],
                                    tile_debug_dense_macs[18:0]);
                            {pending_spike_adds_carry_r,
                             pending_spike_adds_low_r}
                                <= add_u19_to_u24(
                                    status_spike_adds[23:0],
                                    tile_debug_spike_adds[18:0]);
                            state <= ST_COMPUTE_DRAIN;
                        end
                    end
                end

                ST_COMPUTE_DRAIN: begin
                    if (tensor_output_fifo_pop
                            && tensor_output_fifo_head_last) begin
                        state <= ST_COMPUTE_ACCOUNT;
                    end
                end

                ST_COMPUTE_ACCOUNT: begin
                    status_dense_macs <= {
                        status_dense_macs[63:24]
                            + {{39{1'b0}}, pending_dense_macs_carry_r},
                        pending_dense_macs_low_r
                    };
                    status_spike_adds <= {
                        status_spike_adds[63:24]
                            + {{39{1'b0}}, pending_spike_adds_carry_r},
                        pending_spike_adds_low_r
                    };
                    command_done_pending_r <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

    initial begin
        if (MAX_OUTER_ISSUES < 8 || MAX_OUTER_ISSUES > 2048) begin
            $fatal(1, "MAX_OUTER_ISSUES must be in [8,2048]");
        end
        if (MAX_DOT8_ISSUES != MAX_OUTER_ISSUES / 8) begin
            $fatal(1, "MAX_DOT8_ISSUES must equal MAX_OUTER_ISSUES/8");
        end
    end
endmodule

`default_nettype wire
