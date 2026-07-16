`timescale 1ns / 1ps
`default_nettype none

// Converts one DMA stream of descriptor records into ordinary core commands.
// A record is header, command[63:0], command[127:64], then input_count payload
// beats. Header [31:0] is input_count, [62:32] is output_count, and [63] marks
// the final record. External TLAST appears only on the final program input beat;
// the sequencer regenerates the per-command TLAST required by the core.
module neuraedge_descriptor_sequencer (
    input  wire clk,
    input  wire rst_n,

    input  wire program_start,
    input  wire clear_status,
    input  wire [31:0] program_output_beats,
    output logic program_active,
    output logic program_done,
    output logic program_error,
    output logic [31:0] program_records,

    input  wire s_axis_tvalid,
    output logic s_axis_tready,
    input  wire [63:0] s_axis_tdata,
    input  wire s_axis_tlast,

    output logic m_axis_tvalid,
    input  wire m_axis_tready,
    output logic [63:0] m_axis_tdata,
    output logic m_axis_tlast,

    output logic core_command_valid,
    input  wire core_command_ready,
    output logic [127:0] core_command_data,
    input  wire core_command_done,
    input  wire core_status_error,

    output logic core_s_valid,
    input  wire core_s_ready,
    output logic [63:0] core_s_data,
    output logic core_s_last,

    input  wire core_m_valid,
    output logic core_m_ready,
    input  wire [63:0] core_m_data,
    input  wire core_m_last
);
    typedef enum logic [3:0] {
        ST_HEADER = 4'd0,
        ST_COMMAND_LOW = 4'd1,
        ST_COMMAND_HIGH = 4'd2,
        ST_SUBMIT = 4'd3,
        ST_PAYLOAD = 4'd4,
        ST_WAIT = 4'd5,
        ST_ERROR_DRAIN = 4'd6
    } state_t;

    state_t state;
    logic [127:0] descriptor_command_r;
    logic [31:0] input_remaining_r;
    logic input_zero_r;
    logic input_last_r;
    logic [30:0] output_remaining_r;
    logic output_zero_r;
    logic output_last_r;
    logic [31:0] program_output_remaining_r;
    logic program_output_zero_r;
    logic program_output_last_r;
    logic record_final_r;
    logic command_done_seen_r;
    logic command_error_seen_r;
    logic [63:0] input_fifo_data_r [0:1];
    logic input_fifo_last_r [0:1];
    logic input_fifo_write_pointer_r;
    logic input_fifo_read_pointer_r;
    logic [1:0] input_fifo_count_r;
    logic output_buffer_valid_r;
    logic output_buffer_overrun_r;
    logic [63:0] output_buffer_data_r;
    logic output_buffer_last_r;
    // Malformed descriptor detection is quarantined for one cycle.  The
    // pending bit immediately blocks every handshake; on the following edge a
    // local decode clears the FIFO/buffer and either stops or drains input.
    // This separates error detection from a wide multi-register clear cone.
    logic error_pending_r;
    logic error_input_was_last_r;

    wire input_accept = s_axis_tvalid && s_axis_tready;
    wire payload_input_accept = input_accept && state == ST_PAYLOAD;
    wire core_input_accept = core_s_valid && core_s_ready;
    wire output_accept = m_axis_tvalid && m_axis_tready;
    wire core_output_accept = core_m_valid && core_m_ready;
    wire core_output_overrun = core_output_accept
        && (output_zero_r || (output_last_r && output_buffer_valid_r));
    wire record_outputs_complete = output_zero_r;
    wire record_command_complete = command_done_seen_r;

    always_comb begin
        core_command_valid = program_active && !error_pending_r
            && state == ST_SUBMIT;
        core_command_data = descriptor_command_r;

        s_axis_tready = 1'b0;
        core_s_valid = program_active && !error_pending_r
            && (state == ST_PAYLOAD || state == ST_WAIT)
            && input_fifo_count_r != 0;
        core_s_data = input_fifo_data_r[input_fifo_read_pointer_r];
        core_s_last = input_fifo_last_r[input_fifo_read_pointer_r];
        if (program_active && !error_pending_r) begin
            unique case (state)
                ST_HEADER,
                ST_COMMAND_LOW,
                ST_COMMAND_HIGH: s_axis_tready = 1'b1;
                ST_PAYLOAD: s_axis_tready = input_fifo_count_r != 2;
                ST_ERROR_DRAIN: s_axis_tready = 1'b1;
                default: begin end
            endcase
        end

        m_axis_tvalid = program_active && !error_pending_r
            && output_buffer_valid_r && !output_buffer_overrun_r;
        m_axis_tdata = output_buffer_data_r;
        m_axis_tlast = program_output_last_r;
        core_m_ready = program_active && !error_pending_r
            && (state == ST_PAYLOAD || state == ST_WAIT)
            && (!output_buffer_valid_r || m_axis_tready);
    end

`define NE_ENTER_PROGRAM_ERROR(INPUT_WAS_LAST) \
    begin \
        program_error <= 1'b1; \
        error_pending_r <= 1'b1; \
        error_input_was_last_r <= INPUT_WAS_LAST; \
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_HEADER;
            descriptor_command_r <= 128'd0;
            input_remaining_r <= 32'd0;
            input_zero_r <= 1'b1;
            input_last_r <= 1'b0;
            output_remaining_r <= 31'd0;
            output_zero_r <= 1'b1;
            output_last_r <= 1'b0;
            program_output_remaining_r <= 32'd0;
            program_output_zero_r <= 1'b1;
            program_output_last_r <= 1'b0;
            record_final_r <= 1'b0;
            command_done_seen_r <= 1'b0;
            command_error_seen_r <= 1'b0;
            input_fifo_data_r[0] <= 64'd0;
            input_fifo_data_r[1] <= 64'd0;
            input_fifo_last_r[0] <= 1'b0;
            input_fifo_last_r[1] <= 1'b0;
            input_fifo_write_pointer_r <= 1'b0;
            input_fifo_read_pointer_r <= 1'b0;
            input_fifo_count_r <= 2'd0;
            output_buffer_valid_r <= 1'b0;
            output_buffer_overrun_r <= 1'b0;
            output_buffer_data_r <= 64'd0;
            output_buffer_last_r <= 1'b0;
            error_pending_r <= 1'b0;
            error_input_was_last_r <= 1'b0;
            program_active <= 1'b0;
            program_done <= 1'b0;
            if (clear_status && !program_active) program_error <= 1'b0;
            program_error <= 1'b0;
            program_records <= 32'd0;
        end else begin
            program_done <= 1'b0;
            unique case ({payload_input_accept, core_input_accept})
                2'b10: input_fifo_count_r <= input_fifo_count_r + 1'b1;
                2'b01: input_fifo_count_r <= input_fifo_count_r - 1'b1;
                default: begin end
            endcase
            if (payload_input_accept) begin
                input_fifo_data_r[input_fifo_write_pointer_r] <= s_axis_tdata;
                input_fifo_last_r[input_fifo_write_pointer_r]
                    <= input_last_r;
                input_fifo_write_pointer_r <= !input_fifo_write_pointer_r;
            end
            if (core_input_accept) begin
                input_fifo_read_pointer_r <= !input_fifo_read_pointer_r;
            end
            unique case ({core_output_accept, output_accept})
                2'b10: output_buffer_valid_r <= 1'b1;
                2'b01: output_buffer_valid_r <= 1'b0;
                default: begin end
            endcase
            if (core_output_accept) begin
                output_buffer_data_r <= core_m_data;
                output_buffer_last_r <= core_m_last;
                output_buffer_overrun_r <= core_output_overrun;
            end
            if (clear_status && !program_active) program_error <= 1'b0;
            if (program_start) begin
                if (program_active) begin
                    program_error <= 1'b1;
                    program_done <= 1'b1;
                end else begin
                    state <= ST_HEADER;
                    descriptor_command_r <= 128'd0;
                    input_remaining_r <= 32'd0;
                    input_zero_r <= 1'b1;
                    input_last_r <= 1'b0;
                    output_remaining_r <= 31'd0;
                    output_zero_r <= 1'b1;
                    output_last_r <= 1'b0;
                    program_output_remaining_r <= program_output_beats;
                    program_output_zero_r <= program_output_beats == 0;
                    program_output_last_r <= program_output_beats == 1;
                    record_final_r <= 1'b0;
                    command_done_seen_r <= 1'b0;
                    command_error_seen_r <= 1'b0;
                    input_fifo_write_pointer_r <= 1'b0;
                    input_fifo_read_pointer_r <= 1'b0;
                    input_fifo_count_r <= 2'd0;
                    output_buffer_valid_r <= 1'b0;
                    output_buffer_overrun_r <= 1'b0;
                    error_pending_r <= 1'b0;
                    error_input_was_last_r <= 1'b0;
                    program_active <= 1'b1;
                    program_error <= 1'b0;
                    program_records <= 32'd0;
                end
            end

            if (error_pending_r) begin
                error_pending_r <= 1'b0;
                input_fifo_count_r <= 2'd0;
                output_buffer_valid_r <= 1'b0;
                output_buffer_overrun_r <= 1'b0;
                if (error_input_was_last_r) begin
                    program_active <= 1'b0;
                    program_done <= 1'b1;
                    state <= ST_HEADER;
                end else begin
                    state <= ST_ERROR_DRAIN;
                end
            end else if (program_active) begin
                if (output_buffer_valid_r && output_buffer_overrun_r) begin
                    `NE_ENTER_PROGRAM_ERROR(record_final_r)
                end else begin
                    if (core_command_done) begin
                        command_done_seen_r <= 1'b1;
                        command_error_seen_r <= core_status_error;
                    end
                    if (output_accept) begin
                        if (output_buffer_last_r != output_last_r
                                || program_output_zero_r) begin
                            `NE_ENTER_PROGRAM_ERROR(record_final_r)
                        end else begin
                            output_remaining_r <= output_remaining_r - 1'b1;
                            program_output_remaining_r
                                <= program_output_remaining_r - 1'b1;
                            if (output_last_r) begin
                                output_zero_r <= 1'b1;
                                output_last_r <= 1'b0;
                            end else if (output_remaining_r == 2) begin
                                output_last_r <= 1'b1;
                            end
                            if (program_output_last_r) begin
                                program_output_zero_r <= 1'b1;
                                program_output_last_r <= 1'b0;
                            end else if (program_output_remaining_r == 2) begin
                                program_output_last_r <= 1'b1;
                            end
                        end
                    end

                    unique case (state)
                    ST_HEADER: if (input_accept) begin
                        input_remaining_r <= s_axis_tdata[31:0];
                        input_zero_r <= s_axis_tdata[31:0] == 0;
                        input_last_r <= s_axis_tdata[31:0] == 1;
                        output_remaining_r <= s_axis_tdata[62:32];
                        output_zero_r <= s_axis_tdata[62:32] == 0;
                        output_last_r <= s_axis_tdata[62:32] == 1;
                        record_final_r <= s_axis_tdata[63];
                        command_done_seen_r <= 1'b0;
                        command_error_seen_r <= 1'b0;
                        if (s_axis_tlast) `NE_ENTER_PROGRAM_ERROR(1'b1)
                        else state <= ST_COMMAND_LOW;
                    end

                    ST_COMMAND_LOW: if (input_accept) begin
                        descriptor_command_r[63:0] <= s_axis_tdata;
                        if (s_axis_tlast) `NE_ENTER_PROGRAM_ERROR(1'b1)
                        else state <= ST_COMMAND_HIGH;
                    end

                    ST_COMMAND_HIGH: if (input_accept) begin
                        descriptor_command_r[127:64] <= s_axis_tdata;
                        if (s_axis_tlast != (record_final_r
                                && input_zero_r)) begin
                            `NE_ENTER_PROGRAM_ERROR(s_axis_tlast)
                        end else begin
                            state <= ST_SUBMIT;
                        end
                    end

                    ST_SUBMIT: if (core_command_ready) begin
                        program_records <= program_records + 1'b1;
                        if (input_zero_r) state <= ST_WAIT;
                        else state <= ST_PAYLOAD;
                    end

                    ST_PAYLOAD: if (input_accept) begin
                        if (s_axis_tlast != (record_final_r
                                && input_last_r)) begin
                            `NE_ENTER_PROGRAM_ERROR(s_axis_tlast)
                        end else begin
                            input_remaining_r <= input_remaining_r - 1'b1;
                            if (input_last_r) begin
                                input_zero_r <= 1'b1;
                                input_last_r <= 1'b0;
                                state <= ST_WAIT;
                            end else if (input_remaining_r == 2) begin
                                input_last_r <= 1'b1;
                            end
                        end
                    end

                    ST_WAIT: begin
                        if (record_command_complete && record_outputs_complete) begin
                            command_done_seen_r <= 1'b0;
                            command_error_seen_r <= 1'b0;
                            if (command_error_seen_r) begin
                                `NE_ENTER_PROGRAM_ERROR(record_final_r)
                            end else if (record_final_r) begin
                                if (!program_output_zero_r) begin
                                    `NE_ENTER_PROGRAM_ERROR(1'b1)
                                end else begin
                                    program_active <= 1'b0;
                                    program_done <= 1'b1;
                                    state <= ST_HEADER;
                                end
                            end else begin
                                state <= ST_HEADER;
                            end
                        end
                    end

                    ST_ERROR_DRAIN: if (input_accept && s_axis_tlast) begin
                        program_active <= 1'b0;
                        program_done <= 1'b1;
                        state <= ST_HEADER;
                    end

                        default: `NE_ENTER_PROGRAM_ERROR(1'b0)
                    endcase
                end
            end
        end
    end
endmodule

`undef NE_ENTER_PROGRAM_ERROR

`default_nettype wire
