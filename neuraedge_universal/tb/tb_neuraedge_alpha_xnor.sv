`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_alpha_xnor;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [255:0] query_spikes = '0;
    logic [255:0] key_spikes = '0;
    logic [8:0] active_bits = 9'd1;
    logic [8:0] alpha_q8 = 9'd0;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [17:0] score_q8;
    logic [8:0] spike_matches;
    logic [8:0] zero_matches;
    logic config_error;

    int errors = 0;
    int checked = 0;
    logic held_valid = 1'b0;
    logic [17:0] held_score;
    logic [8:0] held_spikes;
    logic [8:0] held_zeros;

    neuraedge_alpha_xnor dut (
        .clk(clk),
        .rst_n(rst_n),
        .input_valid(input_valid),
        .input_ready(input_ready),
        .query_spikes(query_spikes),
        .key_spikes(key_spikes),
        .active_bits(active_bits),
        .alpha_q8(alpha_q8),
        .output_valid(output_valid),
        .output_ready(output_ready),
        .score_q8(score_q8),
        .spike_matches(spike_matches),
        .zero_matches(zero_matches),
        .config_error(config_error)
    );

    always #2.5 clk = ~clk;

    task automatic run_case(
        input logic [255:0] query_value,
        input logic [255:0] key_value,
        input int bit_count,
        input int alpha_value,
        input int stall_cycles
    );
        int expected_spikes;
        int expected_zeros;
        int expected_score;
        begin
            expected_spikes = 0;
            expected_zeros = 0;
            for (int bit_index = 0; bit_index < bit_count; bit_index++) begin
                if (query_value[bit_index] && key_value[bit_index]) expected_spikes++;
                if (!query_value[bit_index] && !key_value[bit_index]) expected_zeros++;
            end
            expected_score = expected_spikes * 256 + expected_zeros * alpha_value;

            while (!input_ready) @(negedge clk);
            @(negedge clk);
            query_spikes = query_value;
            key_spikes = key_value;
            active_bits = 9'(bit_count);
            alpha_q8 = 9'(alpha_value);
            input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 1'b0;

            while (!output_valid) @(negedge clk);
            repeat (stall_cycles) @(negedge clk);
            output_ready = 1'b1;
            @(posedge clk);
            if (config_error || spike_matches !== expected_spikes
                    || zero_matches !== expected_zeros || score_q8 !== expected_score) begin
                errors++;
                $display("ERROR alpha-XNOR bits=%0d alpha=%0d got=(%0d,%0d,%0d) expected=(%0d,%0d,%0d)",
                         bit_count, alpha_value, spike_matches, zero_matches, score_q8,
                         expected_spikes, expected_zeros, expected_score);
            end
            checked++;
            @(negedge clk);
            output_ready = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (held_valid && (!output_valid || score_q8 !== held_score
                || spike_matches !== held_spikes || zero_matches !== held_zeros)) begin
            errors++;
            $display("ERROR alpha-XNOR output changed under backpressure");
        end
        held_valid <= output_valid && !output_ready;
        held_score <= score_q8;
        held_spikes <= spike_matches;
        held_zeros <= zero_matches;
    end

    initial begin
        int seed;
        logic [255:0] query_random;
        logic [255:0] key_random;
        int boundary_bits [0:7];
        int boundary_alpha [0:4];
        seed = 32'h51a7c0de;
        boundary_bits = '{1, 31, 32, 33, 127, 128, 255, 256};
        boundary_alpha = '{0, 1, 127, 255, 256};

        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (int boundary = 0; boundary < 8; boundary++) begin
            query_random = '0;
            key_random = '0;
            for (int word_index = 0; word_index < 8; word_index++) begin
                query_random[word_index*32 +: 32] = $urandom(seed);
                key_random[word_index*32 +: 32] = $urandom(seed);
            end
            for (int alpha_index = 0; alpha_index < 5; alpha_index++) begin
                run_case(query_random, key_random, boundary_bits[boundary],
                         boundary_alpha[alpha_index], (boundary + alpha_index) % 4);
            end
        end

        for (int random_case = 0; random_case < 100; random_case++) begin
            for (int word_index = 0; word_index < 8; word_index++) begin
                query_random[word_index*32 +: 32] = $urandom(seed);
                key_random[word_index*32 +: 32] = $urandom(seed);
            end
            run_case(query_random, key_random, 1 + ($urandom(seed) % 256),
                     $urandom(seed) % 257, random_case % 3);
        end

        // An illegal active length must produce a flagged, deterministic result.
        while (!input_ready) @(negedge clk);
        @(negedge clk);
        active_bits = 9'd0;
        alpha_q8 = 9'd0;
        input_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        input_valid = 1'b0;
        while (!output_valid) @(negedge clk);
        if (!config_error || score_q8 != 0) begin
            errors++;
            $display("ERROR alpha-XNOR illegal configuration was not rejected");
        end
        output_ready = 1'b1;
        @(posedge clk);
        @(negedge clk);
        output_ready = 1'b0;

        if (errors != 0) begin
            $display("tb_neuraedge_alpha_xnor FAILED errors=%0d checked=%0d", errors, checked);
            $fatal(1);
        end else begin
            $display("tb_neuraedge_alpha_xnor PASSED checked=%0d", checked);
            $finish;
        end
    end

    initial begin
        #100_000;
        $fatal(1, "tb_neuraedge_alpha_xnor watchdog timeout");
    end
endmodule

`default_nettype wire
