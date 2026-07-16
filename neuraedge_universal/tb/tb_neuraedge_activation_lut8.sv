`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_activation_lut8;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_table = 1'b0;
    logic load_valid = 1'b0;
    logic load_ready;
    logic [7:0] load_address = 8'd0;
    logic signed [15:0] load_value = 16'sd0;
    logic input_valid = 1'b0;
    logic input_ready;
    logic [63:0] input_indices = 64'd0;
    logic output_valid;
    logic output_ready = 1'b0;
    logic [127:0] output_values;
    logic config_error;
    logic [8:0] loaded_entries;
    int errors = 0;
    int checked = 0;

    neuraedge_activation_lut8 dut (
        .clk(clk), .rst_n(rst_n), .clear_table(clear_table),
        .load_valid(load_valid), .load_ready(load_ready),
        .load_address(load_address), .load_value(load_value),
        .input_valid(input_valid), .input_ready(input_ready),
        .input_indices(input_indices), .output_valid(output_valid),
        .output_ready(output_ready), .output_values(output_values),
        .config_error(config_error), .loaded_entries(loaded_entries)
    );

    always #2.5 clk = ~clk;

    function automatic logic signed [15:0] table_function(input int address);
        int centered;
        int value;
        begin
            centered = address - 128;
            value = centered < 0 ? -(centered * centered) : centered * centered;
            table_function = 16'(value);
        end
    endfunction

    initial begin
        int seed;
        seed = 32'h1a7a2026;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        for (int address = 0; address < 256; address++) begin
            @(negedge clk);
            if (!load_ready) begin
                errors++;
                $display("ERROR LUT load ready address=%0d", address);
            end
            load_address = 8'(address);
            load_value = table_function(address);
            load_valid = 1'b1;
            @(posedge clk);
        end
        @(negedge clk);
        load_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (loaded_entries != 256) begin
            errors++;
            $display("ERROR LUT loaded entries=%0d", loaded_entries);
        end

        // Rewriting an existing address must update its value without changing
        // the unique-entry count. The count commits one cycle after acceptance.
        load_address = 8'd42;
        load_value = table_function(42);
        load_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        load_valid = 1'b0;
        @(posedge clk);
        @(negedge clk);
        if (loaded_entries != 256) begin
            errors++;
            $display("ERROR LUT duplicate changed entries=%0d", loaded_entries);
        end

        for (int test_index = 0; test_index < 100; test_index++) begin
            while (!input_ready) @(negedge clk);
            @(negedge clk);
            for (int lane = 0; lane < 8; lane++) begin
                input_indices[lane*8 +: 8] = $urandom(seed);
            end
            input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            input_valid = 1'b0;
            repeat (test_index % 4) @(negedge clk);
            while (!output_valid) @(negedge clk);
            if (config_error) begin
                errors++;
                $display("ERROR LUT output validity test=%0d", test_index);
            end
            for (int lane = 0; lane < 8; lane++) begin
                if ($signed(output_values[lane*16 +: 16])
                        !== table_function(input_indices[lane*8 +: 8])) begin
                    errors++;
                    $display("ERROR LUT test=%0d lane=%0d address=%0d got=%0d expected=%0d",
                             test_index, lane, input_indices[lane*8 +: 8],
                             $signed(output_values[lane*16 +: 16]),
                             table_function(input_indices[lane*8 +: 8]));
                end
            end
            checked++;
            output_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            output_ready = 1'b0;
        end

        // Clearing the table makes missing entries explicit rather than stale.
        clear_table = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_table = 1'b0;
        input_indices = 64'h0706050403020100;
        input_valid = 1'b1;
        @(posedge clk);
        @(negedge clk);
        input_valid = 1'b0;
        while (!output_valid) @(negedge clk);
        if (!config_error || output_values != 0 || loaded_entries != 0) begin
            errors++;
            $display("ERROR LUT clear/missing-entry behavior");
        end

        if (errors != 0) begin
            $display("tb_neuraedge_activation_lut8 FAILED errors=%0d checked=%0d", errors, checked);
            $finish;
        end else begin
            $display("tb_neuraedge_activation_lut8 PASSED checked=%0d", checked);
            $finish;
        end
    end

    initial begin
        #100_000;
        $fatal(1, "tb_neuraedge_activation_lut8 watchdog timeout");
    end
endmodule

`default_nettype wire
