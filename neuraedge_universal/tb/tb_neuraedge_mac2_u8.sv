`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_mac2_u8;
    logic clk;
    logic rst_n = 1'b0;
    logic enable = 1'b0;
    logic [7:0] activation = '0;
    logic [7:0] weight_lo = '0;
    logic [7:0] weight_hi = '0;
    logic product_valid;
    logic [15:0] product_lo;
    logic [15:0] product_hi;

    int errors = 0;
    int checks = 0;

    neuraedge_mac2_u8 dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .activation(activation),
        .weight_lo(weight_lo),
        .weight_hi(weight_hi),
        .product_valid(product_valid),
        .product_lo(product_lo),
        .product_hi(product_hi)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    task automatic apply_and_check(
        input logic [7:0] a,
        input logic [7:0] w0,
        input logic [7:0] w1
    );
        logic [15:0] expected_lo;
        logic [15:0] expected_hi;
        begin
            expected_lo = a * w0;
            expected_hi = a * w1;
            @(negedge clk);
            enable = 1'b1;
            activation = a;
            weight_lo = w0;
            weight_hi = w1;
            @(posedge clk);
            @(posedge clk);
            @(posedge clk);
            #1;
            checks++;
            if (!product_valid || product_lo !== expected_lo || product_hi !== expected_hi) begin
                errors++;
                if (errors <= 20) begin
                    $display("ERROR a=%0d w0=%0d w1=%0d got=(%0d,%0d) expected=(%0d,%0d)",
                             a, w0, w1, product_lo, product_hi, expected_lo, expected_hi);
                end
            end
        end
    endtask

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1'b1;

        // Exhaust every activation/low-weight combination while sweeping the
        // upper field through a deterministic permutation of all byte values.
        for (int a = 0; a < 256; a++) begin
            for (int w0 = 0; w0 < 256; w0++) begin
                apply_and_check(8'(a), 8'(w0), 8'((w0 * 73 + a * 29 + 17) & 255));
            end
        end

        // Exhaust every activation/high-weight combination independently.
        for (int a = 0; a < 256; a++) begin
            for (int w1 = 0; w1 < 256; w1++) begin
                apply_and_check(8'(a), 8'((w1 * 151 + a * 11 + 3) & 255), 8'(w1));
            end
        end

        @(negedge clk);
        enable = 1'b0;
        @(posedge clk);
        @(posedge clk);
        @(posedge clk);
        #1;
        if (product_valid) begin
            errors++;
            $display("ERROR product_valid did not follow enable low");
        end

        if (errors != 0) begin
            $display("tb_neuraedge_mac2_u8 FAILED checks=%0d errors=%0d", checks, errors);
            $fatal(1);
        end
        $display("tb_neuraedge_mac2_u8 PASSED checks=%0d errors=0", checks);
        $finish;
    end

    initial begin
        #6_000_000;
        $display("ERROR tb_neuraedge_mac2_u8 watchdog timeout");
        $fatal(1);
    end
endmodule

`default_nettype wire
