`timescale 1ns / 1ps
`default_nettype none
`include "rtl/cifar10_spikingformer_head_params.svh"

module tb_cifar10_spikingformer_head_axi_activity_base #(
    parameter bit LEARNING_MODE = 1'b0,
    parameter int NUM_SAMPLES = `CIFAR10_SF_HEAD_NUM_SAMPLES,
    parameter int FEATURE_DIM = `CIFAR10_SF_HEAD_FEATURE_DIM,
    parameter int MAX_SAMPLES = 10000,
    parameter int MAX_FEATURE_WORDS = MAX_SAMPLES * FEATURE_DIM
);
    logic ap_clk = 1'b0;
    logic ap_rst_n = 1'b0;

    logic s_axis_feature_tvalid = 1'b0;
    logic s_axis_feature_tready;
    logic signed [15:0] s_axis_feature_tdata = '0;
    logic s_axis_feature_tlast = 1'b0;

    logic train_enable = 1'b0;
    logic [3:0] train_label = 4'd0;

    logic m_axis_result_tvalid;
    logic m_axis_result_tready = 1'b1;
    logic [31:0] m_axis_result_tdata;

    logic busy;
    logic [31:0] debug_infer_skip_count;
    logic [31:0] debug_train_skip_count;

    logic signed [`CIFAR10_SF_HEAD_FEATURE_BITS-1:0] feature_mem [0:MAX_FEATURE_WORDS-1];
    logic [3:0] label_mem [0:MAX_SAMPLES-1];

    int correct = 0;
    int updates = 0;
    int errors = 0;

    cifar10_spikingformer_head_pynq_axi_stream_top dut (
        .ap_clk(ap_clk),
        .ap_rst_n(ap_rst_n),
        .s_axis_feature_tvalid(s_axis_feature_tvalid),
        .s_axis_feature_tready(s_axis_feature_tready),
        .s_axis_feature_tdata(s_axis_feature_tdata),
        .s_axis_feature_tlast(s_axis_feature_tlast),
        .train_enable(train_enable),
        .train_label(train_label),
        .m_axis_result_tvalid(m_axis_result_tvalid),
        .m_axis_result_tready(m_axis_result_tready),
        .m_axis_result_tdata(m_axis_result_tdata),
        .busy(busy),
        .debug_infer_skip_count(debug_infer_skip_count),
        .debug_train_skip_count(debug_train_skip_count)
    );

    always #5 ap_clk = ~ap_clk;

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

    task automatic send_axis_word(
        input logic signed [15:0] value,
        input logic is_last
    );
        begin
            @(negedge ap_clk);
            s_axis_feature_tdata <= value;
            s_axis_feature_tlast <= is_last;
            s_axis_feature_tvalid <= 1'b1;
            do begin
                @(posedge ap_clk);
            end while (!s_axis_feature_tready);
        end
    endtask

    task automatic send_sample(input int sample_idx);
        logic [3:0] predicted;
        logic update_applied;
        int wait_cycles;
        begin
            train_label <= label_mem[sample_idx];
            train_enable <= LEARNING_MODE;

            for (int f = 0; f < FEATURE_DIM; f++) begin
                send_axis_word(
                    feature_mem[sample_idx * FEATURE_DIM + f],
                    f == FEATURE_DIM - 1
                );
            end

            @(negedge ap_clk);
            s_axis_feature_tvalid <= 1'b0;
            s_axis_feature_tlast <= 1'b0;
            s_axis_feature_tdata <= '0;

            wait_cycles = 0;
            while (!m_axis_result_tvalid && wait_cycles < 20000) begin
                @(posedge ap_clk);
                wait_cycles++;
            end
            if (!m_axis_result_tvalid) begin
                $display("ERROR: timeout waiting for AXI result at sample %0d", sample_idx);
                errors++;
            end else begin
                predicted = m_axis_result_tdata[3:0];
                update_applied = m_axis_result_tdata[4];
                if (predicted == label_mem[sample_idx]) begin
                    correct++;
                end
                if (update_applied) begin
                    updates++;
                end
                if (sample_idx < 8 || (sample_idx % 1000) == 0 || sample_idx == NUM_SAMPLES - 1) begin
                    $display("axi_activity_sample=%0d label=%0d pred=%0d correct=%0d update=%0d infer_skip=%0d train_skip=%0d",
                             sample_idx, label_mem[sample_idx], predicted,
                             predicted == label_mem[sample_idx], update_applied,
                             debug_infer_skip_count, debug_train_skip_count);
                end
                @(posedge ap_clk);
            end
        end
    endtask

    initial begin
        if (NUM_SAMPLES > MAX_SAMPLES) begin
            $display("ERROR: NUM_SAMPLES=%0d exceeds MAX_SAMPLES=%0d", NUM_SAMPLES, MAX_SAMPLES);
            $fatal;
        end

        require_file("rtl/cifar10_spikingformer_head_features.mem");
        require_file("rtl/cifar10_spikingformer_head_labels.mem");
        $readmemh("rtl/cifar10_spikingformer_head_features.mem", feature_mem);
        $readmemh("rtl/cifar10_spikingformer_head_labels.mem", label_mem);

        repeat (8) @(posedge ap_clk);
        ap_rst_n <= 1'b1;
        repeat (4) @(posedge ap_clk);

        for (int s = 0; s < NUM_SAMPLES; s++) begin
            send_sample(s);
        end

        repeat (10) @(posedge ap_clk);
        $display("tb_cifar10_spikingformer_head_axi_activity learning=%0d samples=%0d correct=%0d updates=%0d errors=%0d infer_skip=%0d train_skip=%0d",
                 LEARNING_MODE, NUM_SAMPLES, correct, updates, errors,
                 debug_infer_skip_count, debug_train_skip_count);
        if (errors != 0) begin
            $fatal;
        end
        $finish;
    end
endmodule

module tb_cifar10_spikingformer_head_axi_inference_activity;
    tb_cifar10_spikingformer_head_axi_activity_base #(.LEARNING_MODE(1'b0)) base();
endmodule

module tb_cifar10_spikingformer_head_axi_learning_activity;
    tb_cifar10_spikingformer_head_axi_activity_base #(.LEARNING_MODE(1'b1)) base();
endmodule

`default_nettype wire
