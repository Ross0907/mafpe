`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/operator_fixture_params.svh"

module tb_neuraedge_normalization_units;
    localparam int LN_CASES = `NEURAEDGE_LAYERNORM_CASES;
    localparam int LN_SAMPLES = `NEURAEDGE_LAYERNORM_SAMPLES;
    localparam int SM_CASES = `NEURAEDGE_SOFTMAX_CASES;
    localparam int SM_SAMPLES = `NEURAEDGE_SOFTMAX_SAMPLES;

    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic ln_start_valid = 1'b0;
    logic ln_start_ready;
    logic [11:0] ln_feature_count = 12'd0;
    logic [31:0] ln_epsilon = 32'd1;
    logic ln_sample_valid = 1'b0;
    logic ln_sample_ready;
    logic signed [23:0] ln_sample_value = 24'sd0;
    logic signed [23:0] ln_sample_gamma = 24'sd0;
    logic signed [23:0] ln_sample_beta = 24'sd0;
    logic ln_sample_last = 1'b0;
    logic ln_result_valid;
    logic ln_result_ready = 1'b0;
    logic signed [31:0] ln_result_value;
    logic [11:0] ln_result_index;
    logic ln_result_last;
    logic ln_config_error;
    logic [31:0] ln_cycle_count;
    logic signed [31:0] ln_debug_mean;
    logic [31:0] ln_debug_variance;
    logic [31:0] ln_debug_sigma;
    logic [31:0] ln_debug_reciprocal;

    logic sm_start_valid = 1'b0;
    logic sm_start_ready;
    logic [8:0] sm_item_count = 9'd0;
    logic [4:0] sm_difference_shift = 5'd0;
    logic sm_score_valid = 1'b0;
    logic sm_score_ready;
    logic signed [23:0] sm_score_value = 24'sd0;
    logic sm_score_last = 1'b0;
    logic sm_probability_valid;
    logic sm_probability_ready = 1'b0;
    logic [15:0] sm_probability;
    logic [8:0] sm_probability_index;
    logic sm_probability_last;
    logic sm_config_error;
    logic [31:0] sm_cycle_count;
    logic signed [23:0] sm_debug_maximum;
    logic [23:0] sm_debug_denominator;
    logic [31:0] sm_debug_reciprocal;

    logic [63:0] ln_case_mem [0:LN_CASES-1];
    logic signed [23:0] ln_value_mem [0:LN_SAMPLES-1];
    logic signed [23:0] ln_gamma_mem [0:LN_SAMPLES-1];
    logic signed [23:0] ln_beta_mem [0:LN_SAMPLES-1];
    logic signed [31:0] ln_expected_mem [0:LN_SAMPLES-1];
    logic [127:0] ln_debug_mem [0:LN_CASES-1];
    logic [31:0] sm_case_mem [0:SM_CASES-1];
    logic signed [23:0] sm_score_mem [0:SM_SAMPLES-1];
    logic [15:0] sm_expected_mem [0:SM_SAMPLES-1];
    logic [79:0] sm_debug_mem [0:SM_CASES-1];

    int ln_pointer = 0;
    int sm_pointer = 0;
    int errors = 0;
    int ln_checked = 0;
    int sm_checked = 0;

    neuraedge_integer_layernorm u_layernorm (
        .clk(clk), .rst_n(rst_n),
        .start_valid(ln_start_valid), .start_ready(ln_start_ready),
        .feature_count(ln_feature_count), .epsilon(ln_epsilon),
        .sample_valid(ln_sample_valid), .sample_ready(ln_sample_ready),
        .sample_value(ln_sample_value), .sample_gamma_q15(ln_sample_gamma),
        .sample_beta(ln_sample_beta), .sample_last(ln_sample_last),
        .result_valid(ln_result_valid), .result_ready(ln_result_ready),
        .result_value(ln_result_value), .result_index(ln_result_index),
        .result_last(ln_result_last), .config_error(ln_config_error),
        .cycle_count(ln_cycle_count), .debug_mean(ln_debug_mean),
        .debug_variance(ln_debug_variance), .debug_sigma(ln_debug_sigma),
        .debug_reciprocal_q31(ln_debug_reciprocal)
    );

    neuraedge_log_softmax u_softmax (
        .clk(clk), .rst_n(rst_n),
        .start_valid(sm_start_valid), .start_ready(sm_start_ready),
        .item_count(sm_item_count), .difference_shift(sm_difference_shift),
        .score_valid(sm_score_valid), .score_ready(sm_score_ready),
        .score_value(sm_score_value), .score_last(sm_score_last),
        .probability_valid(sm_probability_valid), .probability_ready(sm_probability_ready),
        .probability_q15(sm_probability), .probability_index(sm_probability_index),
        .probability_last(sm_probability_last), .config_error(sm_config_error),
        .cycle_count(sm_cycle_count), .debug_maximum_score(sm_debug_maximum),
        .debug_denominator(sm_debug_denominator),
        .debug_reciprocal_q31(sm_debug_reciprocal)
    );

    always #2.5 clk = ~clk;

    task automatic run_layernorm_case(input int case_index);
        int count;
        int epsilon_value;
        int received;
        int output_cycle;
        int first_sample;
        begin
            count = ln_case_mem[case_index][11:0];
            epsilon_value = ln_case_mem[case_index][43:12];
            first_sample = ln_pointer;
            while (!ln_start_ready) @(negedge clk);
            @(negedge clk);
            ln_feature_count = 12'(count);
            ln_epsilon = epsilon_value;
            ln_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            ln_start_valid = 1'b0;

            for (int sample = 0; sample < count; sample++) begin
                if (!ln_sample_ready) begin
                    errors++;
                    $display("ERROR LayerNorm sample ready case=%0d sample=%0d", case_index, sample);
                end
                ln_sample_value = ln_value_mem[first_sample + sample];
                ln_sample_gamma = ln_gamma_mem[first_sample + sample];
                ln_sample_beta = ln_beta_mem[first_sample + sample];
                ln_sample_last = sample == count - 1;
                ln_sample_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
            end
            ln_sample_valid = 1'b0;
            ln_sample_last = 1'b0;

            received = 0;
            output_cycle = 0;
            while (received < count && output_cycle < 100000) begin
                ln_result_ready = ((output_cycle + case_index) % 6 != 1);
                @(posedge clk);
                if (ln_result_valid && ln_result_ready) begin
                    if (ln_result_value !== ln_expected_mem[first_sample + received]
                            || ln_result_index !== received
                            || ln_result_last !== (received == count - 1)) begin
                        errors++;
                        if (errors <= 20) begin
                            $display("ERROR LayerNorm case=%0d index=%0d got=%0d expected=%0d rtl_index=%0d last=%0d",
                                     case_index, received, ln_result_value,
                                     ln_expected_mem[first_sample + received], ln_result_index, ln_result_last);
                        end
                    end
                    received++;
                    ln_checked++;
                end
                @(negedge clk);
                output_cycle++;
            end
            ln_result_ready = 1'b0;
            if (received != count || ln_config_error
                    || ln_debug_mean !== ln_debug_mem[case_index][31:0]
                    || ln_debug_variance !== ln_debug_mem[case_index][63:32]
                    || ln_debug_sigma !== ln_debug_mem[case_index][95:64]
                    || ln_debug_reciprocal !== ln_debug_mem[case_index][127:96]) begin
                errors++;
                $display("ERROR LayerNorm completion case=%0d received=%0d/%0d error=%0d debug=(%0d,%0d,%0d,%0d) expected=(%0d,%0d,%0d,%0d)",
                         case_index, received, count, ln_config_error,
                         ln_debug_mean, ln_debug_variance, ln_debug_sigma, ln_debug_reciprocal,
                         $signed(ln_debug_mem[case_index][31:0]), ln_debug_mem[case_index][63:32],
                         ln_debug_mem[case_index][95:64], ln_debug_mem[case_index][127:96]);
            end
            $display("layernorm_case=%0d features=%0d cycles=%0d", case_index, count, ln_cycle_count);
            ln_pointer += count;
        end
    endtask

    task automatic run_softmax_case(input int case_index);
        int count;
        int shift_value;
        int received;
        int output_cycle;
        int first_sample;
        begin
            count = sm_case_mem[case_index][8:0];
            shift_value = sm_case_mem[case_index][13:9];
            first_sample = sm_pointer;
            while (!sm_start_ready) @(negedge clk);
            @(negedge clk);
            sm_item_count = 9'(count);
            sm_difference_shift = 5'(shift_value);
            sm_start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            sm_start_valid = 1'b0;

            for (int sample = 0; sample < count; sample++) begin
                if (!sm_score_ready) begin
                    errors++;
                    $display("ERROR softmax score ready case=%0d sample=%0d", case_index, sample);
                end
                sm_score_value = sm_score_mem[first_sample + sample];
                sm_score_last = sample == count - 1;
                sm_score_valid = 1'b1;
                @(posedge clk);
                @(negedge clk);
            end
            sm_score_valid = 1'b0;
            sm_score_last = 1'b0;

            received = 0;
            output_cycle = 0;
            while (received < count && output_cycle < 100000) begin
                sm_probability_ready = ((output_cycle + case_index) % 5 != 2);
                @(posedge clk);
                if (sm_probability_valid && sm_probability_ready) begin
                    if (sm_probability !== sm_expected_mem[first_sample + received]
                            || sm_probability_index !== received
                            || sm_probability_last !== (received == count - 1)) begin
                        errors++;
                        if (errors <= 20) begin
                            $display("ERROR softmax case=%0d index=%0d got=%0d expected=%0d",
                                     case_index, received, sm_probability,
                                     sm_expected_mem[first_sample + received]);
                        end
                    end
                    received++;
                    sm_checked++;
                end
                @(negedge clk);
                output_cycle++;
            end
            sm_probability_ready = 1'b0;
            if (received != count || sm_config_error
                    || sm_debug_maximum !== sm_debug_mem[case_index][23:0]
                    || sm_debug_denominator !== sm_debug_mem[case_index][47:24]
                    || sm_debug_reciprocal !== sm_debug_mem[case_index][79:48]) begin
                errors++;
                $display("ERROR softmax completion case=%0d received=%0d/%0d error=%0d", case_index,
                         received, count, sm_config_error);
            end
            $display("softmax_case=%0d items=%0d cycles=%0d", case_index, count, sm_cycle_count);
            sm_pointer += count;
        end
    endtask

    initial begin
        $readmemh("tb/fixtures/layernorm_cases.mem", ln_case_mem);
        $readmemh("tb/fixtures/layernorm_values.mem", ln_value_mem);
        $readmemh("tb/fixtures/layernorm_gammas.mem", ln_gamma_mem);
        $readmemh("tb/fixtures/layernorm_betas.mem", ln_beta_mem);
        $readmemh("tb/fixtures/layernorm_expected.mem", ln_expected_mem);
        $readmemh("tb/fixtures/layernorm_debug.mem", ln_debug_mem);
        $readmemh("tb/fixtures/softmax_cases.mem", sm_case_mem);
        $readmemh("tb/fixtures/softmax_scores.mem", sm_score_mem);
        $readmemh("tb/fixtures/softmax_expected.mem", sm_expected_mem);
        $readmemh("tb/fixtures/softmax_debug.mem", sm_debug_mem);

        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (int case_index = 0; case_index < LN_CASES; case_index++) begin
            run_layernorm_case(case_index);
        end
        for (int case_index = 0; case_index < SM_CASES; case_index++) begin
            run_softmax_case(case_index);
        end

        if (ln_pointer != LN_SAMPLES || sm_pointer != SM_SAMPLES
                || ln_checked != LN_SAMPLES || sm_checked != SM_SAMPLES) begin
            errors++;
            $display("ERROR normalization accounting ln=%0d/%0d checked=%0d sm=%0d/%0d checked=%0d",
                     ln_pointer, LN_SAMPLES, ln_checked, sm_pointer, SM_SAMPLES, sm_checked);
        end
        if (errors != 0) begin
            $display("tb_neuraedge_normalization_units FAILED errors=%0d", errors);
            $finish;
        end else begin
            $display("tb_neuraedge_normalization_units PASSED layernorm=%0d softmax=%0d",
                     ln_checked, sm_checked);
            $finish;
        end
    end

    initial begin
        #2_000_000;
        $fatal(1, "tb_neuraedge_normalization_units watchdog timeout");
    end
endmodule

`default_nettype wire
