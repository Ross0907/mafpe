`timescale 1ns / 1ps
`include "rtl/cifar10_transformer_params.svh"

module tb_cifar10_transformer_accelerator;
    parameter int DEFAULT_NUM_SAMPLES = `CIFAR10_NUM_SAMPLES;
    parameter int DEFAULT_TRACE_SAMPLES = `CIFAR10_TRACE_SAMPLES;
    parameter int MAX_SAMPLES = 128;
    parameter int MAX_EVENTS = 500000;
    parameter string EVENTS_FILE = "rtl/cifar10_hw_events.mem";
    parameter string COUNTS_FILE = "rtl/cifar10_hw_counts.mem";
    parameter string LABELS_FILE = "rtl/cifar10_hw_labels.mem";

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic start;
    logic event_valid;
    logic event_ready;
    logic [4:0] event_x;
    logic [4:0] event_y;
    logic [1:0] event_channel;
    logic [7:0] event_value;
    logic frame_done;
    logic busy;
    logic result_valid;
    logic [3:0] predicted_label;
    logic signed [31:0] predicted_score;
    logic [5:0] debug_state;
    logic [7:0] debug_token;
    logic [7:0] debug_key;
    logic [7:0] debug_dim;
    logic signed [31:0] debug_dot;
    logic signed [31:0] debug_cls0;
    logic signed [31:0] debug_cls1;

    logic [19:0] event_mem [0:MAX_EVENTS-1];
    logic [19:0] count_mem [0:MAX_SAMPLES-1];
    logic [3:0] label_mem [0:MAX_SAMPLES-1];

    int num_samples;
    int event_ptr = 0;
    int errors = 0;
    int trace_fh;
    int result_fh;
    int cycle = 0;
    int trace_samples;
    bit trace_enable = 1'b0;
    bit vcd_enabled = 1'b1;

    cifar10_transformer_accelerator dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .event_valid(event_valid),
        .event_ready(event_ready),
        .event_x(event_x),
        .event_y(event_y),
        .event_channel(event_channel),
        .event_value(event_value),
        .frame_done(frame_done),
        .busy(busy),
        .result_valid(result_valid),
        .predicted_label(predicted_label),
        .predicted_score(predicted_score),
        .debug_state(debug_state),
        .debug_token(debug_token),
        .debug_key(debug_key),
        .debug_dim(debug_dim),
        .debug_dot(debug_dot),
        .debug_cls0(debug_cls0),
        .debug_cls1(debug_cls1)
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
        logic [19:0] packed_event;
        int count;
        int waited;
        begin
            trace_enable = (sample_idx < trace_samples);
            if (sample_idx == trace_samples && vcd_enabled) begin
                $dumpoff;
                vcd_enabled = 1'b0;
            end

            count = count_mem[sample_idx];
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            waited = 0;
            while (!event_ready && waited < 5000) begin
                @(posedge clk);
                waited++;
            end
            if (!event_ready) begin
                $display("ERROR: sample %0d never reached event_ready", sample_idx);
                errors++;
            end

            for (int i = 0; i < count; i++) begin
                packed_event = event_mem[event_ptr];
                event_ptr++;
                @(negedge clk);
                event_channel = packed_event[19:18];
                event_x = packed_event[17:13];
                event_y = packed_event[12:8];
                event_value = packed_event[7:0];
                event_valid = 1'b1;
                @(posedge clk);
                #1;
                if (!event_ready) begin
                    $display("ERROR: event_ready dropped during sample %0d event %0d", sample_idx, i);
                    errors++;
                end
            end

            @(negedge clk);
            event_valid = 1'b0;
            frame_done = 1'b1;
            @(negedge clk);
            frame_done = 1'b0;

            waited = 0;
            while (!result_valid && waited < 200000) begin
                @(posedge clk);
                waited++;
            end
            if (!result_valid) begin
                $display("ERROR: sample %0d no result_valid", sample_idx);
                errors++;
            end else begin
                $fwrite(result_fh, "%0d,%0d,%0d,%0d,%0d\n",
                        sample_idx, label_mem[sample_idx], predicted_label, predicted_score,
                        predicted_label == label_mem[sample_idx]);
                $display("sample=%0d label=%0d pred=%0d score=%0d",
                         sample_idx, label_mem[sample_idx], predicted_label, predicted_score);
            end

            waited = 0;
            while (busy && waited < 1000) begin
                @(posedge clk);
                waited++;
            end
            trace_enable = 1'b0;
        end
    endtask

    always @(posedge clk) begin
        if (trace_fh != 0 && trace_enable) begin
            $fwrite(trace_fh, "%0t,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                    $time, cycle, rst_n, debug_state, start, event_valid, event_ready,
                    event_x, event_y, event_channel, event_value, frame_done, result_valid,
                    predicted_label, predicted_score, debug_dot, debug_cls0);
        end
        cycle++;
    end

    initial begin
        require_file(EVENTS_FILE);
        require_file(COUNTS_FILE);
        require_file(LABELS_FILE);
        num_samples = DEFAULT_NUM_SAMPLES;
        trace_samples = DEFAULT_TRACE_SAMPLES;
        if (num_samples > MAX_SAMPLES) begin
            $display("ERROR: NUM_SAMPLES=%0d exceeds MAX_SAMPLES=%0d", num_samples, MAX_SAMPLES);
            $fatal;
        end
        if (trace_samples > num_samples) begin
            trace_samples = num_samples;
        end
        $readmemh(EVENTS_FILE, event_mem);
        $readmemh(COUNTS_FILE, count_mem);
        $readmemh(LABELS_FILE, label_mem);

        $dumpfile("cifar10_transformer_accelerator.vcd");
        $dumpvars(0, tb_cifar10_transformer_accelerator);

        trace_fh = $fopen("cifar10_transformer_trace.csv", "w");
        result_fh = $fopen("cifar10_transformer_results.csv", "w");
        $fwrite(trace_fh, "time_ps,cycle,rst_n,state,start,event_valid,event_ready,event_x,event_y,event_channel,event_value,frame_done,result_valid,predicted_label,predicted_score,debug_dot,debug_cls0\n");
        $fwrite(result_fh, "sample,label,predicted,score,correct\n");

        start = 1'b0;
        event_valid = 1'b0;
        event_x = '0;
        event_y = '0;
        event_channel = '0;
        event_value = '0;
        frame_done = 1'b0;

        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        for (int s = 0; s < num_samples; s++) begin
            send_sample(s);
        end

        repeat (10) @(posedge clk);
        $fclose(trace_fh);
        $fclose(result_fh);

        if (errors != 0) begin
            $display("tb_cifar10_transformer_accelerator FAILED errors=%0d", errors);
            $fatal;
        end

        $display("tb_cifar10_transformer_accelerator PASSED samples=%0d", num_samples);
        $finish;
    end

    initial begin
        #500_000_000;
        $display("ERROR: tb_cifar10_transformer_accelerator watchdog timeout");
        $fatal;
    end
endmodule
