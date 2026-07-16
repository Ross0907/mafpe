`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/tensor_tile_params.svh"

module tb_neuraedge_buffered_tensor_core;
    import neuraedge_command_pkg::*;

    localparam int CASES = `NEURAEDGE_TENSOR_CASES;
    localparam int ISSUES = `NEURAEDGE_TENSOR_ISSUES;
    localparam int RESULTS = `NEURAEDGE_TENSOR_RESULTS;
    localparam int COLUMN_LANES = `NEURAEDGE_COLUMN_LANES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic command_valid = 1'b0;
    logic command_ready;
    logic [127:0] command_data = 128'd0;
    logic command_done;
    logic status_error;
    logic status_busy;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic [63:0] s_axis_tdata = 64'd0;
    logic s_axis_tlast = 1'b0;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b0;
    logic [63:0] m_axis_tdata;
    logic m_axis_tlast;
    logic [11:0] status_loaded_activations;
    logic [11:0] status_loaded_weights;
    logic [11:0] status_active_issues;
    logic [31:0] status_input_beats;
    logic [31:0] status_output_beats;
    logic [31:0] status_compute_cycles;
    logic [31:0] status_skipped_issues;
    logic [63:0] status_dense_macs;
    logic [63:0] status_spike_adds;

    logic [63:0] case_mem [0:CASES-1];
    logic [63:0] activation_mem [0:ISSUES-1];
    logic [COLUMN_LANES*8*8-1:0] weight_mem [0:ISSUES-1];
    logic signed [63:0] expected_mem [0:RESULTS-1];

    int issue_pointer = 0;
    int expected_pointer = 0;
    int checked = 0;
    int errors = 0;
    int cycle = 0;
    logic held_valid = 1'b0;
    logic [63:0] held_data;
    logic held_last;

    neuraedge_buffered_tensor_core dut (
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(command_valid),
        .command_ready(command_ready),
        .command_data(command_data[41:0]),
        .command_done(command_done),
        .status_error(status_error),
        .status_busy(status_busy),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tlast(m_axis_tlast),
        .status_loaded_activations(status_loaded_activations),
        .status_loaded_weights(status_loaded_weights),
        .status_active_issues(status_active_issues),
        .status_input_beats(status_input_beats),
        .status_output_beats(status_output_beats),
        .status_compute_cycles(status_compute_cycles),
        .status_skipped_issues(status_skipped_issues),
        .status_dense_macs(status_dense_macs),
        .status_spike_adds(status_spike_adds)
    );

    always #2.5 clk = ~clk;

    function automatic logic [127:0] make_command(
        input logic [3:0] opcode,
        input int rows,
        input int columns,
        input int issue_count,
        input int activation_zp,
        input int weight_zp
    );
        logic [127:0] value;
        begin
            value = 128'd0;
            value[3:0] = opcode;
            value[7:4] = 4'(rows);
            value[13:8] = 6'(columns);
            value[25:14] = 12'(issue_count);
            value[33:26] = 8'(activation_zp);
            value[41:34] = 8'(weight_zp);
            make_command = value;
        end
    endfunction

    function automatic logic activation_is_zero(
        input logic [63:0] values,
        input logic [7:0] zero_point
    );
        logic is_zero;
        begin
            is_zero = 1'b1;
            for (int lane = 0; lane < 8; lane++) begin
                is_zero = is_zero && values[lane*8 +: 8] == zero_point;
            end
            activation_is_zero = is_zero;
        end
    endfunction

    task automatic submit_command(input logic [127:0] value);
        begin
            while (!command_ready) @(negedge clk);
            @(negedge clk);
            command_data = value;
            command_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            command_valid = 1'b0;
            command_data = 128'd0;
        end
    endtask

    task automatic stream_activation_case(
        input int first_issue,
        input int issue_count
    );
        begin
            for (int issue = 0; issue < issue_count; issue++) begin
                @(negedge clk);
                s_axis_tvalid = 1'b1;
                s_axis_tdata = activation_mem[first_issue + issue];
                s_axis_tlast = (issue == issue_count - 1);
                while (!s_axis_tready) @(negedge clk);
                @(posedge clk);
            end
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tdata = 64'd0;
        end
    endtask

    task automatic stream_weight_case(
        input int first_issue,
        input int issue_count,
        input logic dot8
    );
        int beats_per_issue;
        begin
            beats_per_issue = dot8 ? COLUMN_LANES : COLUMN_LANES / 8;
            for (int issue = 0; issue < issue_count; issue++) begin
                for (int beat = 0; beat < beats_per_issue; beat++) begin
                    @(negedge clk);
                    s_axis_tvalid = 1'b1;
                    s_axis_tdata = weight_mem[first_issue + issue][beat*64 +: 64];
                    s_axis_tlast = issue == issue_count - 1 && beat == beats_per_issue - 1;
                    while (!s_axis_tready) @(negedge clk);
                    @(posedge clk);
                end
            end
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tdata = 64'd0;
        end
    endtask

    task automatic consume_results(
        input int case_index,
        input int result_count
    );
        int received;
        int output_cycle;
        logic signed [63:0] result_extended;
        begin
            received = 0;
            output_cycle = 0;
            while (received < result_count && output_cycle < 20000) begin
                @(negedge clk);
                m_axis_tready = ((output_cycle + case_index) % 7 != 2);
                @(posedge clk);
                if (m_axis_tvalid && m_axis_tready) begin
                    result_extended = {{16{m_axis_tdata[47]}}, m_axis_tdata[47:0]};
                    if (result_extended !== expected_mem[expected_pointer]) begin
                        errors++;
                        if (errors <= 30) begin
                            $display("ERROR buffered case=%0d row=%0d col=%0d got=%0d expected=%0d",
                                     case_index, m_axis_tdata[51:48], m_axis_tdata[57:52],
                                     result_extended, expected_mem[expected_pointer]);
                        end
                    end
                    if (m_axis_tlast !== (received == result_count - 1)) begin
                        errors++;
                        $display("ERROR buffered last mismatch case=%0d result=%0d", case_index, received);
                    end
                    expected_pointer++;
                    received++;
                    checked++;
                end
                output_cycle++;
            end
            @(negedge clk);
            m_axis_tready = 1'b0;
            if (received != result_count) begin
                errors++;
                $display("ERROR buffered output timeout case=%0d got=%0d expected=%0d",
                         case_index, received, result_count);
            end
        end
    endtask

    task automatic run_fixture_case(input int case_index);
        logic [63:0] descriptor;
        logic [3:0] load_weight_opcode;
        logic [3:0] compute_opcode;
        int mode;
        int rows;
        int columns;
        int issue_count;
        int activation_zp;
        int weight_zp;
        int active_count_expected;
        longint unsigned dense_macs_expected;
        longint unsigned spike_adds_expected;
        begin
            descriptor = case_mem[case_index];
            mode = descriptor[1:0];
            rows = descriptor[7:4];
            columns = descriptor[13:8];
            issue_count = descriptor[25:14];
            activation_zp = descriptor[33:26];
            weight_zp = descriptor[41:34];

            load_weight_opcode = mode == MODE_DENSE_DOT8
                ? CMD_LOAD_WEIGHTS_DOT8 : CMD_LOAD_WEIGHTS_OUTER;
            unique case (mode)
                MODE_DENSE_OUTER: compute_opcode = CMD_COMPUTE_DENSE_OUTER;
                MODE_DENSE_DOT8: compute_opcode = CMD_COMPUTE_DENSE_DOT8;
                MODE_BINARY_WEIGHTED: compute_opcode = CMD_COMPUTE_BINARY;
                default: compute_opcode = CMD_COMPUTE_TERNARY;
            endcase

            submit_command(make_command(CMD_CLEAR_STATUS, 0, 0, 0, 0, 0));
            submit_command(make_command(CMD_LOAD_ACTIVATIONS, 0, 0,
                                        issue_count, activation_zp, 0));
            stream_activation_case(issue_pointer, issue_count);
            // Compaction is a registered BRAM write pipeline. Status becomes
            // architecturally visible when the load command commits.
            wait (command_done);
            @(negedge clk);

            active_count_expected = 0;
            dense_macs_expected = 0;
            spike_adds_expected = 0;
            for (int issue = 0; issue < issue_count; issue++) begin
                if (!activation_is_zero(activation_mem[issue_pointer + issue], activation_zp)) begin
                    active_count_expected++;
                    if (mode == MODE_DENSE_OUTER) begin
                        dense_macs_expected += rows * columns;
                    end else if (mode == MODE_DENSE_DOT8) begin
                        dense_macs_expected += 8 * columns;
                    end else begin
                        int active_spikes;
                        active_spikes = 0;
                        for (int row = 0; row < 8; row++) begin
                            if (row < rows
                                    && activation_mem[issue_pointer + issue][row*8 +: 8] != 0) begin
                                active_spikes++;
                            end
                        end
                        spike_adds_expected += active_spikes * columns;
                    end
                end
            end
            if (status_loaded_activations != issue_count
                    || status_active_issues != active_count_expected) begin
                errors++;
                $display("ERROR compaction case=%0d loaded=%0d/%0d active=%0d/%0d",
                         case_index, status_loaded_activations, issue_count,
                         status_active_issues, active_count_expected);
            end

            submit_command(make_command(load_weight_opcode, 0, 0,
                                        issue_count, 0, 0));
            stream_weight_case(issue_pointer, issue_count, mode == MODE_DENSE_DOT8);
            if (status_loaded_weights != issue_count) begin
                errors++;
                $display("ERROR weight count case=%0d loaded=%0d expected=%0d",
                         case_index, status_loaded_weights, issue_count);
            end

            submit_command(make_command(compute_opcode, rows, columns,
                                        issue_count, activation_zp, weight_zp));
            consume_results(case_index, rows * columns);

            // Completion is a registered architectural event.  It follows the
            // final output transfer so status_error is stable in the same cycle.
            wait (command_done);
            @(negedge clk);

            if (!command_done || status_busy || status_error
                    || status_skipped_issues != issue_count - active_count_expected
                    || status_dense_macs != dense_macs_expected
                    || status_spike_adds != spike_adds_expected) begin
                errors++;
                $display("ERROR completion case=%0d done=%0d busy=%0d error=%0d skipped=%0d/%0d dense=%0d/%0d spike=%0d/%0d",
                         case_index, command_done, status_busy, status_error,
                         status_skipped_issues, issue_count - active_count_expected,
                         status_dense_macs, dense_macs_expected,
                         status_spike_adds, spike_adds_expected);
            end
            $display("buffered_case=%0d mode=%0d results=%0d active=%0d/%0d input_beats=%0d compute_cycles=%0d",
                     case_index, mode, rows * columns, active_count_expected,
                     issue_count, status_input_beats, status_compute_cycles);
            issue_pointer += issue_count;
        end
    endtask

    task automatic check_counter_add(
        input logic [63:0] value,
        input logic [18:0] increment
    );
        logic [63:0] actual;
        logic [63:0] expected;
        logic [24:0] low_sum;
        begin
            low_sum = dut.add_u19_to_u24(value[23:0], increment);
            actual = {
                value[63:24] + {{39{1'b0}}, low_sum[24]},
                low_sum[23:0]
            };
            expected = value + {{45{1'b0}}, increment};
            if (actual !== expected) begin
                errors++;
                $display("ERROR counter add value=%h increment=%h actual=%h expected=%h",
                         value, increment, actual, expected);
            end
        end
    endtask

    task automatic verify_counter_adder;
        logic [63:0] random_value;
        logic [18:0] random_increment;
        int unsigned random_state;
        begin
            check_counter_add(64'd0, 19'd0);
            check_counter_add(64'd0, 19'h7ffff);
            check_counter_add(64'h0000000000ffffff, 19'd1);
            check_counter_add(64'h00000000ffffffff, 19'd1);
            check_counter_add(64'h0000ffffffffffff, 19'd1);
            check_counter_add(64'h00ffffffffffffff, 19'd1);
            check_counter_add(64'hffffffffffffffff, 19'h7ffff);
            check_counter_add(64'h55fffffffffffeaa, 19'h10155);

            random_state = 32'h4e455552;
            for (int vector_index = 0; vector_index < 1024;
                    vector_index++) begin
                random_state = random_state * 32'd1664525 + 32'd1013904223;
                random_value[31:0] = random_state;
                random_state = random_state * 32'd1664525 + 32'd1013904223;
                random_value[63:32] = random_state;
                random_state = random_state * 32'd1664525 + 32'd1013904223;
                random_increment = random_state[18:0];
                check_counter_add(random_value, random_increment);
            end
        end
    endtask

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (held_valid && (!m_axis_tvalid || m_axis_tdata !== held_data
                || m_axis_tlast !== held_last)) begin
            errors++;
            $display("ERROR buffered output changed while backpressured");
        end
        held_valid <= m_axis_tvalid && !m_axis_tready;
        held_data <= m_axis_tdata;
        held_last <= m_axis_tlast;
    end

    initial begin
        $readmemh("tb/fixtures/tensor_cases.mem", case_mem);
        $readmemh("tb/fixtures/tensor_activations.mem", activation_mem);
        $readmemh("tb/fixtures/tensor_weights.mem", weight_mem);
        $readmemh("tb/fixtures/tensor_expected.mem", expected_mem);

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        verify_counter_adder();

        for (int case_index = 0; case_index < CASES; case_index++) begin
            run_fixture_case(case_index);
        end

        if (issue_pointer != ISSUES || expected_pointer != RESULTS || checked != RESULTS) begin
            errors++;
            $display("ERROR buffered accounting issue=%0d/%0d expected=%0d/%0d checked=%0d",
                     issue_pointer, ISSUES, expected_pointer, RESULTS, checked);
        end

        if (errors != 0) begin
            $display("tb_neuraedge_buffered_tensor_core FAILED errors=%0d checked=%0d", errors, checked);
            $fatal(1);
        end else begin
            $display("tb_neuraedge_buffered_tensor_core PASSED cases=%0d checked=%0d", CASES, checked);
        end
        $finish;
    end

    initial begin
        #200_000;
        $display("ERROR buffered core watchdog state=%0d busy=%0d cmd_ready=%0d s_ready=%0d m_valid=%0d",
                 dut.state, status_busy, command_ready, s_axis_tready, m_axis_tvalid);
        $fatal(1);
    end
endmodule

`default_nettype wire
