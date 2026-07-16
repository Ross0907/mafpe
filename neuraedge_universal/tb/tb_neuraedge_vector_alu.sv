`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/operator_fixture_params.svh"

module tb_neuraedge_vector_alu;
    localparam int CASES = `NEURAEDGE_VECTOR_CASES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [3:0] operation = 4'd0;
    logic [255:0] operand_a = '0;
    logic [255:0] operand_b = '0;
    logic signed [15:0] multiplier_q15 = 16'sd0;
    logic [4:0] right_shift = 5'd0;
    logic signed [15:0] output_zero_point = 16'sd0;
    logic signed [31:0] threshold = 32'sd0;
    logic [31:0] surrogate_window = 32'd0;
    logic signed [31:0] clamp_minimum = 32'sh80000000;
    logic signed [31:0] clamp_maximum = 32'sh7fffffff;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [255:0] result;
    logic config_error;

    logic [191:0] command_mem [0:CASES-1];
    logic [255:0] operand_a_mem [0:CASES-1];
    logic [255:0] operand_b_mem [0:CASES-1];
    logic [255:0] expected_mem [0:CASES-1];
    int errors = 0;
    int checked = 0;
    logic held_valid = 1'b0;
    logic [255:0] held_result;

    neuraedge_vector_alu dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .operation(operation),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .multiplier_q15(multiplier_q15),
        .right_shift(right_shift),
        .output_zero_point(output_zero_point),
        .threshold(threshold),
        .surrogate_window(surrogate_window),
        .clamp_minimum(clamp_minimum),
        .clamp_maximum(clamp_maximum),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .result(result),
        .config_error(config_error)
    );

    always #2.5 clk = ~clk;

    always @(posedge clk) begin
        if (held_valid && (!output_valid || result !== held_result)) begin
            errors++;
            $display("ERROR vector ALU output changed while stalled");
        end
        held_valid <= output_valid && !output_ready;
        held_result <= result;
    end

    initial begin
        $readmemh("tb/fixtures/vector_commands.mem", command_mem);
        $readmemh("tb/fixtures/vector_a.mem", operand_a_mem);
        $readmemh("tb/fixtures/vector_b.mem", operand_b_mem);
        $readmemh("tb/fixtures/vector_expected.mem", expected_mem);

        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (int case_index = 0; case_index < CASES; case_index++) begin
            logic [191:0] command;
            command = command_mem[case_index];
            while (!input_ready) @(negedge clk);
            @(negedge clk);
            operation = command[3:0];
            multiplier_q15 = command[19:4];
            right_shift = command[24:20];
            output_zero_point = command[40:25];
            threshold = command[72:41];
            surrogate_window = command[104:73];
            clamp_minimum = command[136:105];
            clamp_maximum = command[168:137];
            operand_a = operand_a_mem[case_index];
            operand_b = operand_b_mem[case_index];
            input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 1'b0;

            while (!output_valid) @(negedge clk);
            repeat (case_index % 5) @(negedge clk);
            if (!output_valid || config_error || result !== expected_mem[case_index]) begin
                errors++;
                if (errors <= 20) begin
                    $display("ERROR vector case=%0d op=%0d config=%0d", case_index, operation, config_error);
                    for (int lane = 0; lane < 8; lane++) begin
                        if (result[lane*32 +: 32] !== expected_mem[case_index][lane*32 +: 32]) begin
                            $display("  lane=%0d got=%0d expected=%0d", lane,
                                     $signed(result[lane*32 +: 32]),
                                     $signed(expected_mem[case_index][lane*32 +: 32]));
                        end
                    end
                end
            end
            checked++;
            output_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            output_ready = 1'b0;
        end

        // Unsupported operations are visible as deterministic configuration errors.
        while (!input_ready) @(negedge clk);
        @(negedge clk);
        operation = 4'hf;
        input_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        input_valid = 1'b0;
        while (!output_valid) @(negedge clk);
        if (!output_valid || !config_error || result != 0) begin
            errors++;
            $display("ERROR vector ALU invalid opcode handling");
        end
        output_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);

        if (errors != 0) begin
            $display("tb_neuraedge_vector_alu FAILED errors=%0d checked=%0d", errors, checked);
            $finish;
        end else begin
            $display("tb_neuraedge_vector_alu PASSED checked=%0d", checked);
            $finish;
        end
    end

    initial begin
        #500_000;
        $fatal(1, "tb_neuraedge_vector_alu watchdog timeout");
    end
endmodule

`default_nettype wire
