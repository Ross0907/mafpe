`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_lif_bank;
    localparam int LANES = 32;
    localparam int MEMBRANE_BITS = 24;
    localparam int CURRENT_BITS = 16;
    localparam int MAXIMUM_MEMBRANE = (1 << 23) - 1;
    localparam int MINIMUM_MEMBRANE = -(1 << 23);

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_state = 1'b0;
    logic clear_counters = 1'b0;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [LANES*CURRENT_BITS-1:0] current_vector = '0;
    logic [4:0] leak_shift = 5'd2;
    logic signed [MEMBRANE_BITS-1:0] threshold = 24'sd100;
    logic signed [MEMBRANE_BITS-1:0] reset_value = 24'sd0;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [LANES-1:0] spikes;
    logic [LANES*MEMBRANE_BITS-1:0] membrane_vector;
    logic [LANES*MEMBRANE_BITS-1:0] pre_reset_membrane_vector;
    logic [31:0] timestep_count;
    logic [63:0] total_spike_count;

    integer reference_membrane [0:LANES-1];
    int reference_timesteps = 0;
    int reference_spikes = 0;
    int errors = 0;
    int checked = 0;

    neuraedge_lif_bank dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear_state(clear_state),
        .clear_counters(clear_counters),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .current_vector(current_vector),
        .leak_shift(leak_shift),
        .threshold(threshold),
        .reset_value(reset_value),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .spikes(spikes),
        .membrane_vector(membrane_vector),
        .pre_reset_membrane_vector(pre_reset_membrane_vector),
        .timestep_count(timestep_count),
        .total_spike_count(total_spike_count)
    );

    always #2.5 clk = ~clk;

    function automatic int saturate_reference(input longint signed value);
        begin
            if (value > MAXIMUM_MEMBRANE) saturate_reference = MAXIMUM_MEMBRANE;
            else if (value < MINIMUM_MEMBRANE) saturate_reference = MINIMUM_MEMBRANE;
            else saturate_reference = int'(value);
        end
    endfunction

    task automatic clear_reference_and_dut(input int reset_level);
        begin
            reset_value = 24'(reset_level);
            @(negedge clk);
            clear_state = 1'b1;
            @(posedge clk);
            for (int lane = 0; lane < LANES; lane++) reference_membrane[lane] = reset_level;
            reference_timesteps = 0;
            @(negedge clk);
            clear_state = 1'b0;
        end
    endtask

    task automatic run_step(input int stall_cycles);
        logic [LANES-1:0] expected_spikes;
        integer expected_membrane [0:LANES-1];
        integer expected_pre_reset_membrane [0:LANES-1];
        begin
            expected_spikes = '0;
            for (int lane = 0; lane < LANES; lane++) begin
                int current_value;
                longint signed integrated;
                current_value = $signed(current_vector[lane*CURRENT_BITS +: CURRENT_BITS]);
                integrated = reference_membrane[lane]
                    - (reference_membrane[lane] >>> leak_shift) + current_value;
                expected_membrane[lane] = saturate_reference(integrated);
                expected_pre_reset_membrane[lane] = expected_membrane[lane];
                if (expected_membrane[lane] >= $signed(threshold)) begin
                    expected_spikes[lane] = 1'b1;
                    expected_membrane[lane] = $signed(reset_value);
                end
            end

            while (!input_ready) @(negedge clk);
            @(negedge clk);
            input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 1'b0;
            while (!output_valid) @(negedge clk);
            repeat (stall_cycles) begin
                if (input_ready) begin
                    errors++;
                    $display("ERROR LIF accepted a new step while output was stalled");
                end
                @(negedge clk);
            end

            if (!output_valid || spikes !== expected_spikes) begin
                errors++;
                $display("ERROR LIF spike mismatch step=%0d got=%h expected=%h",
                         reference_timesteps, spikes, expected_spikes);
            end
            for (int lane = 0; lane < LANES; lane++) begin
                logic signed [MEMBRANE_BITS-1:0] got_membrane;
                logic signed [MEMBRANE_BITS-1:0] got_pre_reset_membrane;
                got_membrane = membrane_vector[lane*MEMBRANE_BITS +: MEMBRANE_BITS];
                got_pre_reset_membrane = pre_reset_membrane_vector[
                    lane*MEMBRANE_BITS +: MEMBRANE_BITS];
                if ($signed(got_membrane) !== expected_membrane[lane]) begin
                    errors++;
                    if (errors <= 20) begin
                        $display("ERROR LIF membrane step=%0d lane=%0d got=%0d expected=%0d",
                                 reference_timesteps, lane, $signed(got_membrane), expected_membrane[lane]);
                    end
                end
                if ($signed(got_pre_reset_membrane)
                        !== expected_pre_reset_membrane[lane]) begin
                    errors++;
                    if (errors <= 20) begin
                        $display("ERROR LIF pre-reset membrane step=%0d lane=%0d got=%0d expected=%0d",
                                 reference_timesteps, lane,
                                 $signed(got_pre_reset_membrane),
                                 expected_pre_reset_membrane[lane]);
                    end
                end
                reference_membrane[lane] = expected_membrane[lane];
                reference_spikes += expected_spikes[lane];
            end
            reference_timesteps++;

            output_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            output_ready = 1'b0;
            if (timestep_count != reference_timesteps || total_spike_count != reference_spikes) begin
                errors++;
                $display("ERROR LIF counters got=(%0d,%0d) expected=(%0d,%0d)",
                         timestep_count, total_spike_count, reference_timesteps, reference_spikes);
            end
            checked++;
        end
    endtask

    initial begin
        int seed;
        seed = 32'h1f1f2026;
        for (int lane = 0; lane < LANES; lane++) reference_membrane[lane] = 0;

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        clear_reference_and_dut(0);

        for (int step = 0; step < 100; step++) begin
            for (int lane = 0; lane < LANES; lane++) begin
                current_vector[lane*CURRENT_BITS +: CURRENT_BITS] =
                    16'($signed(($urandom(seed) % 241) - 120));
            end
            leak_shift = 5'(1 + step % 5);
            threshold = 24'sd100;
            run_step(step % 4);
        end

        // Explicit positive saturation boundary.
        clear_reference_and_dut(MAXIMUM_MEMBRANE - 4);
        leak_shift = 5'd31;
        threshold = 24'sd8388607;
        current_vector = '0;
        current_vector[CURRENT_BITS-1:0] = 16'sd32767;
        run_step(2);

        @(negedge clk);
        clear_counters = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_counters = 1'b0;
        reference_spikes = 0;
        if (total_spike_count != 0) begin
            errors++;
            $display("ERROR LIF spike counter did not clear independently");
        end

        if (errors != 0) begin
            $display("tb_neuraedge_lif_bank FAILED errors=%0d checked=%0d", errors, checked);
            $finish;
        end else begin
            $display("tb_neuraedge_lif_bank PASSED checked=%0d timesteps=%0d spikes=%0d",
                     checked, reference_timesteps, reference_spikes);
            $finish;
        end
    end

    initial begin
        #200_000;
        $fatal(1, "tb_neuraedge_lif_bank watchdog timeout");
    end
endmodule

`default_nettype wire
