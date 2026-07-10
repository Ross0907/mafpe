`timescale 1ns / 1ps
`include "rtl/cifar10_spikingformer_head_params.svh"

module tb_cifar10_spikingformer_head_onchip_learning;
    parameter int NUM_SAMPLES = `CIFAR10_SF_HEAD_NUM_SAMPLES;
    parameter int TRACE_SAMPLES = `CIFAR10_SF_HEAD_TRACE_SAMPLES;
    parameter int FEATURE_DIM = `CIFAR10_SF_HEAD_FEATURE_DIM;
    parameter int MAX_SAMPLES = 10000;
    parameter int MAX_FEATURE_WORDS = MAX_SAMPLES * FEATURE_DIM;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start;
    logic feature_valid;
    logic feature_ready;
    logic signed [`CIFAR10_SF_HEAD_FEATURE_BITS-1:0] feature_value;
    logic frame_done;
    logic train_enable;
    logic [3:0] train_label;
    logic busy;
    logic result_valid;
    logic [3:0] predicted_label;
    logic signed [`CIFAR10_SF_HEAD_ACC_BITS-1:0] predicted_score;
    logic train_update_valid;
    logic train_update_applied;
    logic [5:0] debug_state;
    logic [15:0] debug_feature;
    logic [7:0] debug_class;
    logic [31:0] debug_update_count;
    logic [31:0] debug_infer_skip_count;
    logic [31:0] debug_train_skip_count;

    logic signed [`CIFAR10_SF_HEAD_FEATURE_BITS-1:0] feature_mem [0:MAX_FEATURE_WORDS-1];
    logic [3:0] label_mem [0:MAX_SAMPLES-1];

    int feature_ptr = 0;
    int correct = 0;
    int errors = 0;
    int updates = 0;
    int cycle = 0;
    int trace_fh;
    int result_fh;
    int weight_fh;
    int bias_fh;
    bit trace_enable = 1'b0;
    bit vcd_enabled = 1'b1;

    cifar10_spikingformer_head_accelerator #(
        .ENABLE_ONCHIP_LEARNING(1),
        .LEARN_SHIFT(14),
        .BIAS_LEARN_DELTA(1),
        .LEARN_MARGIN_Q(64'sd268435456)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .feature_valid(feature_valid),
        .feature_ready(feature_ready),
        .feature_value(feature_value),
        .frame_done(frame_done),
        .train_enable(train_enable),
        .train_label(train_label),
        .busy(busy),
        .result_valid(result_valid),
        .predicted_label(predicted_label),
        .predicted_score(predicted_score),
        .train_update_valid(train_update_valid),
        .train_update_applied(train_update_applied),
        .debug_state(debug_state),
        .debug_feature(debug_feature),
        .debug_class(debug_class),
        .debug_update_count(debug_update_count),
        .debug_infer_skip_count(debug_infer_skip_count),
        .debug_train_skip_count(debug_train_skip_count)
    );

    always #5 clk = ~clk;

    task automatic require_file(input string path);
        int fh;
        begin
            fh = $fopen(path, "r");
            if (fh == 0) begin
                $display("ERROR: missing required file %s", path);
                $fatal;
            end
            $fclose(fh);
        end
    endtask

    task automatic send_sample(input int sample_idx);
        int waited;
        bit update_seen;
        begin
            trace_enable = (sample_idx < TRACE_SAMPLES);
            if (sample_idx == TRACE_SAMPLES && vcd_enabled) begin
                $dumpoff;
                vcd_enabled = 1'b0;
            end

            @(negedge clk);
            train_label = label_mem[sample_idx];
            train_enable = 1'b1;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            waited = 0;
            while (!feature_ready && waited < 10000) begin
                @(posedge clk);
                waited++;
            end
            if (!feature_ready) begin
                $display("ERROR: sample %0d never reached feature_ready", sample_idx);
                errors++;
            end

            for (int f = 0; f < FEATURE_DIM; f++) begin
                @(negedge clk);
                feature_value = feature_mem[feature_ptr];
                feature_ptr++;
                feature_valid = 1'b1;
                @(posedge clk);
                #1;
                if (f < FEATURE_DIM - 1 && !feature_ready) begin
                    $display("ERROR: feature_ready dropped during sample %0d feature %0d", sample_idx, f);
                    errors++;
                end
            end

            @(negedge clk);
            feature_valid = 1'b0;
            frame_done = 1'b1;
            @(negedge clk);
            frame_done = 1'b0;

            waited = 0;
            update_seen = 1'b0;
            while (!result_valid && waited < 200000) begin
                @(posedge clk);
                if (train_update_applied) begin
                    update_seen = 1'b1;
                end
                waited++;
            end
            if (!result_valid) begin
                $display("ERROR: sample %0d no result_valid", sample_idx);
                errors++;
            end else begin
                if (predicted_label == label_mem[sample_idx]) begin
                    correct++;
                end
                if (update_seen) begin
                    updates++;
                end
                $fwrite(result_fh, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                        sample_idx, label_mem[sample_idx], predicted_label, predicted_score,
                        predicted_label == label_mem[sample_idx], update_seen, debug_update_count,
                        debug_infer_skip_count, debug_train_skip_count);
                for (int c = 0; c < 10; c++) begin
                    $fwrite(result_fh, ",%0d", $signed(dut.class_score_mem[c]));
                end
                $fwrite(result_fh, "\n");
                if (sample_idx < TRACE_SAMPLES || (sample_idx % 1000) == 0 || sample_idx == NUM_SAMPLES - 1) begin
                    $display("learn_sample=%0d label=%0d pred=%0d correct=%0d update=%0d",
                             sample_idx, label_mem[sample_idx], predicted_label,
                             predicted_label == label_mem[sample_idx], update_seen);
                end
            end

            waited = 0;
            while (busy && waited < 1000) begin
                @(posedge clk);
                waited++;
            end
            train_enable = 1'b0;
            trace_enable = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (trace_fh != 0 && trace_enable) begin
            $fwrite(trace_fh, "%0t,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    $time, cycle, rst_n, debug_state, start, feature_valid, feature_ready,
                    debug_feature, debug_class, result_valid, predicted_label,
                    train_update_applied, debug_update_count, debug_infer_skip_count, debug_train_skip_count);
        end
        cycle++;
    end

    initial begin
        if (NUM_SAMPLES > MAX_SAMPLES) begin
            $display("ERROR: NUM_SAMPLES=%0d exceeds MAX_SAMPLES=%0d", NUM_SAMPLES, MAX_SAMPLES);
            $fatal;
        end
        require_file("rtl/cifar10_spikingformer_head_features.mem");
        require_file("rtl/cifar10_spikingformer_head_weights.mem");
        require_file("rtl/cifar10_spikingformer_head_bias.mem");
        require_file("rtl/cifar10_spikingformer_head_labels.mem");
        $readmemh("rtl/cifar10_spikingformer_head_features.mem", feature_mem);
        $readmemh("rtl/cifar10_spikingformer_head_labels.mem", label_mem);

        $dumpfile("cifar10_spikingformer_head_onchip_learning.vcd");
        $dumpvars(0, tb_cifar10_spikingformer_head_onchip_learning);
        trace_fh = $fopen("cifar10_spikingformer_head_onchip_learning_trace.csv", "w");
        result_fh = $fopen("cifar10_spikingformer_head_onchip_learning_results.csv", "w");
        $fwrite(trace_fh, "time_ps,cycle,rst_n,state,start,feature_valid,feature_ready,feature_idx,class_idx,result_valid,predicted_label,train_update_applied,update_count,infer_skip_count,train_skip_count\n");
        $fwrite(result_fh, "sample,label,predicted,score,correct,update_applied,update_count,infer_skip_count,train_skip_count");
        for (int c = 0; c < 10; c++) begin
            $fwrite(result_fh, ",score%0d", c);
        end
        $fwrite(result_fh, "\n");

        start = 1'b0;
        feature_valid = 1'b0;
        feature_value = '0;
        frame_done = 1'b0;
        train_enable = 1'b0;
        train_label = 4'd0;

        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        for (int s = 0; s < NUM_SAMPLES; s++) begin
            send_sample(s);
        end

        repeat (10) @(posedge clk);
        weight_fh = $fopen("cifar10_spikingformer_head_onchip_learning_weights_after.mem", "w");
        bias_fh = $fopen("cifar10_spikingformer_head_onchip_learning_bias_after.mem", "w");
        for (int c = 0; c < 10; c++) begin
            for (int f = 0; f < FEATURE_DIM; f++) begin
                $fwrite(weight_fh, "%0d\n", $signed(dut.weight_mem[c * FEATURE_DIM + f]));
            end
            $fwrite(bias_fh, "%0d\n", $signed(dut.bias_mem[c]));
        end
        $fclose(weight_fh);
        $fclose(bias_fh);
        $fclose(trace_fh);
        $fclose(result_fh);
        $display("tb_cifar10_spikingformer_head_onchip_learning samples=%0d correct=%0d updates=%0d errors=%0d", NUM_SAMPLES, correct, updates, errors);
        if (errors != 0) begin
            $fatal;
        end
        $finish;
    end

    initial begin
        #2_000_000_000;
        $display("ERROR: tb_cifar10_spikingformer_head_onchip_learning watchdog timeout");
        $fatal;
    end
endmodule
