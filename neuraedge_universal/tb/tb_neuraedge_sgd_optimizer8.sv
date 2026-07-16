`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/operator_fixture_params.svh"

module tb_neuraedge_sgd_optimizer8;
    localparam int CASES = `NEURAEDGE_OPTIMIZER_CASES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_counters = 1'b0;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [127:0] weight_vector = '0;
    logic [191:0] gradient_vector = '0;
    logic [191:0] velocity_vector = '0;
    logic [17:0] learning_rate_q16 = 18'd0;
    logic [16:0] momentum_q16 = 17'd0;
    logic [16:0] weight_decay_q16 = 17'd0;
    logic [4:0] update_right_shift = 5'd16;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [127:0] updated_weight_vector;
    logic [191:0] updated_velocity_vector;
    logic config_error;
    logic [31:0] update_count;

    logic [63:0] parameter_mem [0:CASES-1];
    logic [127:0] weight_mem [0:CASES-1];
    logic [191:0] gradient_mem [0:CASES-1];
    logic [191:0] velocity_mem [0:CASES-1];
    logic [127:0] expected_weight_mem [0:CASES-1];
    logic [191:0] expected_velocity_mem [0:CASES-1];
    int errors = 0;
    int checked = 0;
    logic held_valid = 1'b0;
    logic [127:0] held_weight;
    logic [191:0] held_velocity;

    neuraedge_sgd_optimizer8 dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear_counters(clear_counters),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .weight_vector(weight_vector),
        .gradient_vector(gradient_vector),
        .velocity_vector(velocity_vector),
        .learning_rate_q16(learning_rate_q16),
        .momentum_q16(momentum_q16),
        .weight_decay_q16(weight_decay_q16),
        .update_right_shift(update_right_shift),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .updated_weight_vector(updated_weight_vector),
        .updated_velocity_vector(updated_velocity_vector),
        .config_error(config_error),
        .update_count(update_count)
    );

    always #2.5 clk = ~clk;

    always @(posedge clk) begin
        if (held_valid && (!output_valid || updated_weight_vector !== held_weight
                || updated_velocity_vector !== held_velocity)) begin
            errors++;
            $display("ERROR optimizer output changed while stalled");
        end
        held_valid <= output_valid && !output_ready;
        held_weight <= updated_weight_vector;
        held_velocity <= updated_velocity_vector;
    end

    initial begin
        $readmemh("tb/fixtures/optimizer_parameters.mem", parameter_mem);
        $readmemh("tb/fixtures/optimizer_weights.mem", weight_mem);
        $readmemh("tb/fixtures/optimizer_gradients.mem", gradient_mem);
        $readmemh("tb/fixtures/optimizer_velocities.mem", velocity_mem);
        $readmemh("tb/fixtures/optimizer_expected_weights.mem", expected_weight_mem);
        $readmemh("tb/fixtures/optimizer_expected_velocities.mem", expected_velocity_mem);

        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (int case_index = 0; case_index < CASES; case_index++) begin
            while (!input_ready) @(negedge clk);
            @(negedge clk);
            learning_rate_q16 = parameter_mem[case_index][17:0];
            momentum_q16 = parameter_mem[case_index][34:18];
            weight_decay_q16 = parameter_mem[case_index][51:35];
            update_right_shift = parameter_mem[case_index][56:52];
            weight_vector = weight_mem[case_index];
            gradient_vector = gradient_mem[case_index];
            velocity_vector = velocity_mem[case_index];
            input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 1'b0;

            while (!output_valid) @(negedge clk);
            repeat (case_index % 4) @(negedge clk);
            if (config_error || updated_weight_vector !== expected_weight_mem[case_index]
                    || updated_velocity_vector !== expected_velocity_mem[case_index]) begin
                errors++;
                if (errors <= 20) begin
                    $display("ERROR optimizer case=%0d config=%0d", case_index, config_error);
                    for (int lane = 0; lane < 8; lane++) begin
                        if (updated_weight_vector[lane*16 +: 16]
                                !== expected_weight_mem[case_index][lane*16 +: 16]) begin
                            $display("  weight lane=%0d got=%0d expected=%0d", lane,
                                     $signed(updated_weight_vector[lane*16 +: 16]),
                                     $signed(expected_weight_mem[case_index][lane*16 +: 16]));
                        end
                        if (updated_velocity_vector[lane*24 +: 24]
                                !== expected_velocity_mem[case_index][lane*24 +: 24]) begin
                            $display("  velocity lane=%0d got=%0d expected=%0d", lane,
                                     $signed(updated_velocity_vector[lane*24 +: 24]),
                                     $signed(expected_velocity_mem[case_index][lane*24 +: 24]));
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

        if (update_count != CASES * 8) begin
            errors++;
            $display("ERROR optimizer update count got=%0d expected=%0d", update_count, CASES * 8);
        end

        @(negedge clk);
        clear_counters = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_counters = 1'b0;
        if (update_count != 0) begin
            errors++;
            $display("ERROR optimizer update counter did not clear independently");
        end

        if (errors != 0) begin
            $display("tb_neuraedge_sgd_optimizer8 FAILED errors=%0d checked=%0d", errors, checked);
            $finish;
        end else begin
            $display("tb_neuraedge_sgd_optimizer8 PASSED checked=%0d updates=%0d", checked, update_count);
            $finish;
        end
    end

    initial begin
        #200_000;
        $fatal(1, "tb_neuraedge_sgd_optimizer8 watchdog timeout");
    end
endmodule

`default_nettype wire
