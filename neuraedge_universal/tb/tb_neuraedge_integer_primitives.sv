`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_integer_primitives;
    logic clk = 1'b0;
    logic rst_n = 1'b0;

    logic div_input_valid = 1'b0;
    logic div_input_ready;
    logic [63:0] div_numerator = 64'd0;
    logic [31:0] div_denominator = 32'd1;
    logic div_output_valid;
    logic div_output_ready = 1'b0;
    logic [63:0] div_quotient;
    logic [31:0] div_remainder;
    logic divide_by_zero;

    logic sqrt_input_valid = 1'b0;
    logic sqrt_input_ready;
    logic [63:0] sqrt_radicand = 64'd0;
    logic sqrt_output_valid;
    logic sqrt_output_ready = 1'b0;
    logic [31:0] sqrt_root;
    logic [32:0] sqrt_remainder;

    int errors = 0;
    int div_checked = 0;
    int sqrt_checked = 0;

    neuraedge_udivider u_divider (
        .clk(clk), .rst_n(rst_n),
        .input_valid(div_input_valid), .input_ready(div_input_ready),
        .numerator(div_numerator), .denominator(div_denominator),
        .output_valid(div_output_valid), .output_ready(div_output_ready),
        .quotient(div_quotient), .remainder(div_remainder),
        .divide_by_zero(divide_by_zero)
    );

    neuraedge_isqrt64 u_sqrt (
        .clk(clk), .rst_n(rst_n),
        .input_valid(sqrt_input_valid), .input_ready(sqrt_input_ready),
        .radicand(sqrt_radicand), .output_valid(sqrt_output_valid),
        .output_ready(sqrt_output_ready), .root(sqrt_root),
        .remainder(sqrt_remainder)
    );

    always #2.5 clk = ~clk;

    task automatic check_division(input logic [63:0] n, input logic [31:0] d);
        logic [63:0] expected_q;
        logic [31:0] expected_r;
        begin
            expected_q = d == 0 ? 64'hffffffffffffffff : n / d;
            expected_r = d == 0 ? n[31:0] : n % d;
            while (!div_input_ready) @(negedge clk);
            @(negedge clk);
            div_numerator = n;
            div_denominator = d;
            div_input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            div_input_valid = 1'b0;
            while (!div_output_valid) @(negedge clk);
            repeat (div_checked % 3) @(negedge clk);
            if (div_quotient !== expected_q || div_remainder !== expected_r
                    || divide_by_zero !== (d == 0)) begin
                errors++;
                $display("ERROR divider n=%0d d=%0d got=%0d r=%0d expected=%0d r=%0d",
                         n, d, div_quotient, div_remainder, expected_q, expected_r);
            end
            div_output_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            div_output_ready = 1'b0;
            div_checked++;
        end
    endtask

    task automatic check_sqrt(input logic [63:0] value);
        logic [63:0] expected_root;
        begin
            expected_root = 0;
            // The testbench oracle uses binary search, independent of the
            // restoring square-root datapath in the DUT.
            begin
                logic [63:0] low;
                logic [63:0] high;
                low = 0;
                high = 64'h0000000100000000;
                while (low + 1 < high) begin
                    logic [63:0] middle;
                    middle = (low + high) >> 1;
                    if (middle <= value / middle) low = middle;
                    else high = middle;
                end
                expected_root = low;
            end
            while (!sqrt_input_ready) @(negedge clk);
            @(negedge clk);
            sqrt_radicand = value;
            sqrt_input_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            sqrt_input_valid = 1'b0;
            while (!sqrt_output_valid) @(negedge clk);
            repeat (sqrt_checked % 4) @(negedge clk);
            if (sqrt_root !== expected_root[31:0]
                    || sqrt_remainder !== value - expected_root * expected_root) begin
                errors++;
                $display("ERROR sqrt value=%0d got=%0d rem=%0d expected=%0d rem=%0d",
                         value, sqrt_root, sqrt_remainder, expected_root,
                         value - expected_root * expected_root);
            end
            sqrt_output_ready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            sqrt_output_ready = 1'b0;
            sqrt_checked++;
        end
    endtask

    initial begin
        int seed;
        logic [63:0] random_value;
        seed = 32'hd171de42;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        check_division(0, 1);
        check_division(1, 1);
        check_division(64'hffffffffffffffff, 32'hffffffff);
        check_division(64'd123456789012345, 32'd97);
        check_division(1234, 0);
        for (int test_index = 0; test_index < 100; test_index++) begin
            random_value = {$urandom(seed), $urandom(seed)};
            check_division(random_value, 1 + ($urandom(seed) % 32'hfffffffe));
        end

        check_sqrt(0);
        check_sqrt(1);
        check_sqrt(2);
        check_sqrt(3);
        check_sqrt(4);
        check_sqrt(64'hffffffffffffffff);
        for (int test_index = 0; test_index < 100; test_index++) begin
            random_value = {$urandom(seed), $urandom(seed)};
            check_sqrt(random_value);
        end

        if (errors != 0) begin
            $display("tb_neuraedge_integer_primitives FAILED errors=%0d", errors);
            $finish;
        end else begin
            $display("tb_neuraedge_integer_primitives PASSED divisions=%0d square_roots=%0d",
                     div_checked, sqrt_checked);
            $finish;
        end
    end

    initial begin
        #500_000;
        $fatal(1, "tb_neuraedge_integer_primitives watchdog timeout");
    end
endmodule

`default_nettype wire
