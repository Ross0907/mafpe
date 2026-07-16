`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/tensor_tile_params.svh"

module tb_neuraedge_tensor_tile;
    localparam int CASES = `NEURAEDGE_TENSOR_CASES;
    localparam int ISSUES = `NEURAEDGE_TENSOR_ISSUES;
    localparam int RESULTS = `NEURAEDGE_TENSOR_RESULTS;
    localparam int ROW_LANES = 8;
    localparam int COLUMN_LANES = `NEURAEDGE_COLUMN_LANES;

    logic clk;
    logic rst_n = 1'b0;
    logic start = 1'b0;
    logic [1:0] compute_mode = 2'd0;
    logic [3:0] active_rows = 4'd1;
    logic [5:0] active_columns = 6'd1;
    logic [7:0] activation_zero_point = 8'd0;
    logic [7:0] weight_zero_point = 8'd0;
    logic config_error;
    logic issue_valid = 1'b0;
    logic issue_ready;
    logic [63:0] activation_vector = 64'd0;
    logic [COLUMN_LANES*8-1:0] outer_weight_vector = '0;
    logic [COLUMN_LANES*8*8-1:0] dot_weight_vector = '0;
    logic issue_last = 1'b0;
    logic busy;
    logic result_valid;
    logic result_ready = 1'b0;
    logic signed [47:0] result_value;
    logic [3:0] result_row;
    logic [5:0] result_column;
    logic result_last;
    logic [31:0] debug_accepted_issues;
    logic [31:0] debug_skipped_issues;
    logic [31:0] debug_retired_issues;
    logic [63:0] debug_dense_macs;
    logic [63:0] debug_spike_adds;

    logic [63:0] case_mem [0:CASES-1];
    logic [63:0] activation_mem [0:ISSUES-1];
    logic [COLUMN_LANES*8*8-1:0] weight_mem [0:ISSUES-1];
    logic signed [63:0] expected_mem [0:RESULTS-1];

    int issue_pointer = 0;
    int expected_pointer = 0;
    int errors = 0;
    int checked = 0;
    int cycle = 0;
    logic stalled_valid = 1'b0;
    logic signed [47:0] stalled_value;
    logic [3:0] stalled_row;
    logic [5:0] stalled_column;
    logic stalled_last;

    neuraedge_tensor_tile dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .compute_mode(compute_mode),
        .active_rows(active_rows),
        .active_columns(active_columns),
        .activation_zero_point(activation_zero_point),
        .weight_zero_point(weight_zero_point),
        .config_error(config_error),
        .issue_valid(issue_valid),
        .issue_ready(issue_ready),
        .activation_vector(activation_vector),
        .outer_weight_vector(outer_weight_vector),
        .dot_weight_vector(dot_weight_vector),
        .issue_last(issue_last),
        .busy(busy),
        .result_valid(result_valid),
        .result_ready(result_ready),
        .result_value(result_value),
        .result_row(result_row),
        .result_column(result_column),
        .result_last(result_last),
        .debug_accepted_issues(debug_accepted_issues),
        .debug_skipped_issues(debug_skipped_issues),
        .debug_retired_issues(debug_retired_issues),
        .debug_dense_macs(debug_dense_macs),
        .debug_spike_adds(debug_spike_adds)
    );

    initial clk = 1'b0;
    always #2.5 clk = ~clk;

    task automatic run_case(input int case_index);
        logic [63:0] command;
        int case_issues;
        int case_results;
        int received;
        int watchdog;
        int output_cycle;
        longint unsigned expected_dense_macs;
        longint unsigned expected_spike_adds;
        logic signed [63:0] result_extended;
        begin
            command = case_mem[case_index];
            compute_mode = command[1:0];
            active_rows = command[7:4];
            active_columns = command[13:8];
            case_issues = command[25:14];
            activation_zero_point = command[33:26];
            weight_zero_point = command[41:34];
            // Cast before multiplying. Without these casts, SystemVerilog keeps
            // a full tile product can wrap before assignment to an integer.
            case_results = int'(active_rows) * int'(active_columns);
            expected_dense_macs = 0;
            expected_spike_adds = 0;

            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
            if (config_error) begin
                errors++;
                $display("ERROR case=%0d unexpectedly reported config_error", case_index);
            end

            for (int issue = 0; issue < case_issues; issue++) begin
                while (!issue_ready) @(negedge clk);
                activation_vector = activation_mem[issue_pointer];
                if (compute_mode == 2'd0 || compute_mode == 2'd1) begin
                    logic issue_nonzero;
                    issue_nonzero = 1'b0;
                    for (int row = 0; row < ROW_LANES; row++) begin
                        issue_nonzero = issue_nonzero
                            || activation_mem[issue_pointer][row*8 +: 8]
                                != activation_zero_point;
                    end
                    if (issue_nonzero) begin
                        expected_dense_macs += compute_mode == 2'd0
                            ? int'(active_rows) * int'(active_columns)
                            : ROW_LANES * int'(active_columns);
                    end
                end else begin
                    int active_spikes;
                    active_spikes = 0;
                    for (int row = 0; row < ROW_LANES; row++) begin
                        if (row < active_rows
                                && activation_mem[issue_pointer][row*8 +: 8] != 0) begin
                            active_spikes++;
                        end
                    end
                    expected_spike_adds += active_spikes * int'(active_columns);
                end
                dot_weight_vector = weight_mem[issue_pointer];
                outer_weight_vector = weight_mem[issue_pointer][COLUMN_LANES*8-1:0];
                issue_last = (issue == case_issues - 1);
                issue_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
                issue_valid = 1'b0;
                issue_last = 1'b0;
                issue_pointer++;
            end

            received = 0;
            watchdog = 0;
            output_cycle = 0;
            if (case_index == 0) begin
                // The first fixture is the maximum 8x16 tile. Keep READY low
                // until all 128 records are resident, proving that compute can
                // finish without a downstream consumer and without overflow.
                result_ready = 1'b0;
                while (int'(dut.result_fifo_count_r)
                            + int'(dut.result_stage_valid_r) < case_results
                        && watchdog < 2000) begin
                    @(negedge clk);
                    watchdog++;
                end
                if (int'(dut.result_fifo_count_r)
                            + int'(dut.result_stage_valid_r) != case_results
                        || !dut.result_fifo_nonempty_r
                        || !dut.result_fifo_has_space_r
                        || !dut.result_stage_valid_r
                        || dut.p5_valid_r) begin
                    errors++;
                    $display("ERROR maximum result buffer did not fill exactly memory_count=%0d stage_valid=%0d nonempty=%0d space=%0d p5_valid=%0d",
                             dut.result_fifo_count_r, dut.result_stage_valid_r,
                             dut.result_fifo_nonempty_r,
                             dut.result_fifo_has_space_r, dut.p5_valid_r);
                end
                watchdog = 0;
            end
            while (received < case_results && watchdog < 20000) begin
                @(negedge clk);
                // Drive READY independently of the result count. Tying stalls to
                // `received` can deadlock a ready/valid test when a stalled count
                // selects the same false value forever.
                result_ready = ((output_cycle + case_index) % 5 != 1);
                // READY is now stable before the active edge. Sample the result
                // on the same rising edge on which the DUT commits the transfer.
                @(posedge clk);
                if (result_valid && result_ready) begin
                    result_extended = {{16{result_value[47]}}, result_value};
                    if (result_extended !== expected_mem[expected_pointer]) begin
                        errors++;
                        if (errors <= 30) begin
                            $display("ERROR case=%0d row=%0d col=%0d got=%0d expected=%0d",
                                     case_index, result_row, result_column,
                                     result_extended, expected_mem[expected_pointer]);
                        end
                    end
                    if (result_last !== (received == case_results - 1)) begin
                        errors++;
                        $display("ERROR case=%0d result_last mismatch at result=%0d", case_index, received);
                    end
                    expected_pointer++;
                    received++;
                    checked++;
                end
                output_cycle++;
                watchdog++;
            end
            @(negedge clk);
            result_ready = 1'b0;
            if (received != case_results) begin
                errors++;
                $display("ERROR case=%0d output timeout received=%0d expected=%0d", case_index, received, case_results);
            end
            while (busy) @(negedge clk);
            if (debug_dense_macs !== expected_dense_macs
                    || debug_spike_adds !== expected_spike_adds) begin
                errors++;
                $display("ERROR case=%0d counter mismatch dense=%0d/%0d spike=%0d/%0d",
                         case_index, debug_dense_macs, expected_dense_macs,
                         debug_spike_adds, expected_spike_adds);
            end
            $display("case=%0d mode=%0d rows=%0d columns=%0d issues=%0d skips=%0d dense_macs=%0d spike_adds=%0d",
                     case_index, compute_mode, active_rows, active_columns, case_issues,
                     debug_skipped_issues, debug_dense_macs, debug_spike_adds);
        end
    endtask

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (stalled_valid) begin
            if (!result_valid || result_value !== stalled_value || result_row !== stalled_row
                    || result_column !== stalled_column || result_last !== stalled_last) begin
                errors++;
                $display("ERROR result changed while backpressured");
            end
        end
        stalled_valid <= result_valid && !result_ready;
        stalled_value <= result_value;
        stalled_row <= result_row;
        stalled_column <= result_column;
        stalled_last <= result_last;
    end

    initial begin
        $readmemh("tb/fixtures/tensor_cases.mem", case_mem);
        $readmemh("tb/fixtures/tensor_activations.mem", activation_mem);
        $readmemh("tb/fixtures/tensor_weights.mem", weight_mem);
        $readmemh("tb/fixtures/tensor_expected.mem", expected_mem);
        // The complete numerical regression deliberately runs without a VCD:
        // 128 MAC instances make an unrestricted trace needlessly enormous.
        // A bounded, purpose-built waveform test generates the review VCD.

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (int case_index = 0; case_index < CASES; case_index++) begin
            run_case(case_index);
        end

        // Synchronous reset recovery is checked after all numerical cases so it cannot
        // conceal a fixture mismatch.
        compute_mode = 2'd0;
        active_rows = 4'd1;
        active_columns = 6'd1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        if (!busy) begin
            errors++;
            $display("ERROR reset-recovery setup did not enter busy state");
        end
        rst_n = 1'b0;
        repeat (2) @(negedge clk);
        if (busy || result_valid || issue_ready) begin
            errors++;
            $display("ERROR external reset did not return tile to idle outputs");
        end
        rst_n = 1'b1;

        if (issue_pointer != ISSUES || expected_pointer != RESULTS || checked != RESULTS) begin
            errors++;
            $display("ERROR fixture accounting issue_ptr=%0d/%0d result_ptr=%0d/%0d checked=%0d",
                     issue_pointer, ISSUES, expected_pointer, RESULTS, checked);
        end
        if (errors != 0) begin
            $display("tb_neuraedge_tensor_tile FAILED errors=%0d checked=%0d", errors, checked);
            $fatal(1);
        end
        $display("tb_neuraedge_tensor_tile PASSED cases=%0d checked=%0d", CASES, checked);
        $finish;
    end

    initial begin
        #100_000;
        $display("ERROR tb_neuraedge_tensor_tile watchdog timeout state=%0d busy=%0d issue_ready=%0d result_valid=%0d result_ready=%0d accepted=%0d retired=%0d mac_issued=%0d mac_retired=%0d target=%0d",
                 dut.state, busy, issue_ready, result_valid, result_ready,
                 debug_accepted_issues, debug_retired_issues,
                 dut.mac_issue_count, dut.mac_retire_count, dut.target_mac_retires);
        $fatal(1);
    end
endmodule

`default_nettype wire
