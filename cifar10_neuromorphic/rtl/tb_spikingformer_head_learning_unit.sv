`timescale 1ns / 1ps

module tb_spikingformer_head_learning_unit;
    localparam int FEATURE_DIM = 4;
    localparam int NUM_CLASSES = 3;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start;
    logic feature_valid;
    logic feature_ready;
    logic signed [15:0] feature_value;
    logic frame_done;
    logic train_enable;
    logic [3:0] train_label;
    logic busy;
    logic result_valid;
    logic [3:0] predicted_label;
    logic signed [63:0] predicted_score;
    logic train_update_valid;
    logic train_update_applied;
    logic [5:0] debug_state;
    logic [15:0] debug_feature;
    logic [7:0] debug_class;
    logic [31:0] debug_update_count;
    logic [31:0] debug_infer_skip_count;
    logic [31:0] debug_train_skip_count;

    logic signed [15:0] x [0:FEATURE_DIM-1];
    int errors = 0;

    cifar10_spikingformer_head_accelerator #(
        .FEATURE_DIM(FEATURE_DIM),
        .NUM_CLASSES(NUM_CLASSES),
        .FEATURE_BITS(16),
        .WEIGHT_BITS(16),
        .ACC_BITS(64),
        .ENABLE_ONCHIP_LEARNING(1),
        .INFER_SKIP_ABS_THRESH(0),
        .LEARN_SHIFT(1),
        .BIAS_LEARN_DELTA(3),
        .LEARN_MARGIN_Q(64'sd1000000)
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

    task automatic expect_i16(input string name, input int got, input int expected);
        begin
            if (got !== expected) begin
                $display("ERROR: %s got=%0d expected=%0d", name, got, expected);
                errors++;
            end
        end
    endtask

    initial begin
        x[0] = 16'sd4;
        x[1] = -16'sd2;
        x[2] = 16'sd6;
        x[3] = -16'sd8;

        start = 1'b0;
        feature_valid = 1'b0;
        feature_value = 16'sd0;
        frame_done = 1'b0;
        train_enable = 1'b0;
        train_label = 4'd1;

        repeat (8) @(negedge clk);
        for (int i = 0; i < NUM_CLASSES * FEATURE_DIM; i++) begin
            dut.weight_mem[i] = 16'sd0;
        end
        dut.bias_mem[0] = 64'sd0;
        dut.bias_mem[1] = 64'sd0;
        dut.bias_mem[2] = 64'sd20;
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        @(negedge clk);
        train_enable = 1'b1;
        train_label = 4'd1;
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        while (!feature_ready) @(posedge clk);
        for (int f = 0; f < FEATURE_DIM; f++) begin
            @(negedge clk);
            feature_value = x[f];
            feature_valid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk);
        feature_valid = 1'b0;
        frame_done = 1'b1;
        @(negedge clk);
        frame_done = 1'b0;

        while (!result_valid) @(posedge clk);
        if (predicted_label !== 4'd2) begin
            $display("ERROR: predicted_label got=%0d expected=2", predicted_label);
            errors++;
        end
        if (!train_update_applied && debug_update_count != 32'd1) begin
            $display("ERROR: update did not apply");
            errors++;
        end

        while (busy) @(posedge clk);
        expect_i16("class1 weight0", dut.weight_mem[1 * FEATURE_DIM + 0], 2);
        expect_i16("class1 weight1", dut.weight_mem[1 * FEATURE_DIM + 1], -1);
        expect_i16("class1 weight2", dut.weight_mem[1 * FEATURE_DIM + 2], 3);
        expect_i16("class1 weight3", dut.weight_mem[1 * FEATURE_DIM + 3], -4);
        expect_i16("class2 weight0", dut.weight_mem[2 * FEATURE_DIM + 0], -2);
        expect_i16("class2 weight1", dut.weight_mem[2 * FEATURE_DIM + 1], 1);
        expect_i16("class2 weight2", dut.weight_mem[2 * FEATURE_DIM + 2], -3);
        expect_i16("class2 weight3", dut.weight_mem[2 * FEATURE_DIM + 3], 4);
        if ($signed(dut.bias_mem[1]) !== 64'sd3 || $signed(dut.bias_mem[2]) !== 64'sd17) begin
            $display("ERROR: bias update got class1=%0d class2=%0d", $signed(dut.bias_mem[1]), $signed(dut.bias_mem[2]));
            errors++;
        end

        $display("tb_spikingformer_head_learning_unit errors=%0d", errors);
        if (errors != 0) $fatal;
        $finish;
    end
endmodule
