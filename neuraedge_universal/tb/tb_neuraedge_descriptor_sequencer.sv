`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_descriptor_sequencer;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic program_start = 1'b0;
    logic clear_status = 1'b0;
    logic [31:0] program_output_beats = 32'd0;
    logic program_active;
    logic program_done;
    logic program_error;
    logic [31:0] program_records;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic [63:0] s_axis_tdata = 64'd0;
    logic s_axis_tlast = 1'b0;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b1;
    logic [63:0] m_axis_tdata;
    logic m_axis_tlast;
    logic core_command_valid;
    logic core_command_ready;
    logic [127:0] core_command_data;
    logic core_command_done = 1'b0;
    logic core_status_error = 1'b0;
    logic core_s_valid;
    logic core_s_ready;
    logic [63:0] core_s_data;
    logic core_s_last;
    logic core_m_valid;
    logic core_m_ready;
    logic [63:0] core_m_data;
    logic core_m_last;

    typedef enum logic [1:0] {
        CORE_IDLE = 2'd0,
        CORE_INPUT = 2'd1,
        CORE_OUTPUT = 2'd2,
        CORE_DONE = 2'd3
    } core_state_t;
    core_state_t core_state = CORE_IDLE;
    logic core_output_index = 1'b0;
    int errors = 0;
    int output_count = 0;
    int program_output_index = 0;
    int expected_program_outputs = 2;
    int input_count = 0;

    neuraedge_descriptor_sequencer dut (
        .clk(clk), .rst_n(rst_n),
        .program_start(program_start),
        .clear_status(clear_status),
        .program_output_beats(program_output_beats),
        .program_active(program_active), .program_done(program_done),
        .program_error(program_error), .program_records(program_records),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast),
        .core_command_valid(core_command_valid),
        .core_command_ready(core_command_ready),
        .core_command_data(core_command_data),
        .core_command_done(core_command_done),
        .core_status_error(core_status_error),
        .core_s_valid(core_s_valid), .core_s_ready(core_s_ready),
        .core_s_data(core_s_data), .core_s_last(core_s_last),
        .core_m_valid(core_m_valid), .core_m_ready(core_m_ready),
        .core_m_data(core_m_data), .core_m_last(core_m_last)
    );

    always #2.5 clk = ~clk;

    always_comb begin
        core_command_ready = core_state == CORE_IDLE;
        core_s_ready = core_state == CORE_INPUT;
        core_m_valid = core_state == CORE_OUTPUT;
        core_m_data = core_output_index ? 64'h2222 : 64'h1111;
        core_m_last = core_output_index;
    end

    always @(posedge clk) begin
        core_command_done <= 1'b0;
        if (!rst_n) begin
            core_state <= CORE_IDLE;
            core_output_index <= 1'b0;
        end else begin
            unique case (core_state)
                CORE_IDLE: if (core_command_valid && core_command_ready) begin
                    unique case (core_command_data[3:0])
                        4'h1: core_state <= CORE_INPUT;
                        4'h4: begin
                            core_output_index <= 1'b0;
                            core_state <= CORE_OUTPUT;
                        end
                        default: core_state <= CORE_DONE;
                    endcase
                end
                CORE_INPUT: if (core_s_valid && core_s_ready) begin
                    input_count <= input_count + 1;
                    if ((input_count == 0 && core_s_last)
                            || (input_count == 1 && !core_s_last)) begin
                        errors <= errors + 1;
                    end
                    if (core_s_last) core_state <= CORE_DONE;
                end
                CORE_OUTPUT: if (core_m_valid && core_m_ready) begin
                    if (core_output_index) core_state <= CORE_DONE;
                    else core_output_index <= 1'b1;
                end
                CORE_DONE: begin
                    core_command_done <= 1'b1;
                    core_state <= CORE_IDLE;
                end
                default: core_state <= CORE_IDLE;
            endcase
        end
    end

    always @(posedge clk) begin
        if (m_axis_tvalid && m_axis_tready) begin
            if (m_axis_tdata !== (program_output_index == 0 ? 64'h1111 : 64'h2222)
                    || m_axis_tlast !== (program_output_index
                        == expected_program_outputs - 1)) begin
                errors <= errors + 1;
                $display("ERROR output=%0d data=%h last=%0d",
                         program_output_index, m_axis_tdata, m_axis_tlast);
            end
            output_count <= output_count + 1;
            program_output_index <= program_output_index + 1;
        end
    end

    task automatic send_word(input logic [63:0] data, input logic last);
        begin
            while (!s_axis_tready) @(negedge clk);
            s_axis_tdata = data;
            s_axis_tlast = last;
            s_axis_tvalid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        @(negedge clk);
        program_output_beats = 2;
        program_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        program_start = 1'b0;

        // Load command: two payload beats, no output, not the final record.
        send_word({1'b0, 31'd0, 32'd2}, 1'b0);
        send_word(64'h1, 1'b0);
        send_word(64'd0, 1'b0);
        send_word(64'habc, 1'b0);
        send_word(64'hdef, 1'b0);

        // Compute command: no input and two outputs. It is deliberately not
        // the final input record; global output count must still generate TLAST.
        send_word({1'b0, 31'd2, 32'd0}, 1'b0);
        send_word(64'h4, 1'b0);
        send_word(64'd0, 1'b0);

        // A trailing no-output command proves output TLAST does not depend on
        // which descriptor carries the final input marker.
        send_word({1'b1, 31'd0, 32'd0}, 1'b0);
        send_word(64'hf, 1'b0);
        send_word(64'd0, 1'b1);

        wait (program_done);
        @(negedge clk);
        if (program_error || program_active || program_records != 3
                || input_count != 2 || output_count != 2) begin
            errors++;
            $display("ERROR final active=%0d error=%0d records=%0d input=%0d output=%0d",
                     program_active, program_error, program_records,
                     input_count, output_count);
        end

        // Declare one output for a core command that produces two. The first
        // beat is legal and gets TLAST; the second must be discarded while the
        // descriptor sequencer reports an overrun instead of deadlocking.
        @(negedge clk);
        program_output_index = 0;
        expected_program_outputs = 1;
        program_output_beats = 1;
        program_start = 1'b1;
        @(posedge clk);
        @(negedge clk);
        program_start = 1'b0;
        send_word({1'b1, 31'd1, 32'd0}, 1'b0);
        send_word(64'h4, 1'b0);
        send_word(64'd0, 1'b1);
        wait (program_done);
        @(negedge clk);
        if (!program_error || program_active || program_records != 1
                || output_count != 3) begin
            errors++;
            $display("ERROR overrun active=%0d error=%0d records=%0d output=%0d",
                     program_active, program_error, program_records, output_count);
        end
        if (errors != 0) begin
            $display("tb_neuraedge_descriptor_sequencer FAILED errors=%0d", errors);
        end else begin
            $display("tb_neuraedge_descriptor_sequencer PASSED records=%0d", program_records);
        end
        $finish;
    end

    initial begin
        #200_000;
        $fatal(1, "descriptor sequencer watchdog timeout");
    end
endmodule

`default_nettype wire
