`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_pynq_descriptor;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [7:0] awaddr = 8'd0;
    logic awvalid = 1'b0;
    wire awready;
    logic [31:0] wdata = 32'd0;
    logic [3:0] wstrb = 4'hf;
    logic wvalid = 1'b0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    logic bready = 1'b0;
    logic [7:0] araddr = 8'd0;
    logic arvalid = 1'b0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    logic rready = 1'b0;
    logic s_valid = 1'b0;
    wire s_ready;
    logic [63:0] s_data = 64'd0;
    logic [7:0] s_keep = 8'hff;
    logic s_last = 1'b0;
    wire m_valid;
    logic m_ready = 1'b0;
    wire [63:0] m_data;
    wire [7:0] m_keep;
    wire m_last;
    wire interrupt;
    int errors = 0;

    neuraedge_pynq_axi_top dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid),
        .s_axi_wready(wready), .s_axi_bresp(bresp), .s_axi_bvalid(bvalid),
        .s_axi_bready(bready), .s_axi_araddr(araddr), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .s_axis_tvalid(s_valid), .s_axis_tready(s_ready), .s_axis_tdata(s_data),
        .s_axis_tkeep(s_keep), .s_axis_tlast(s_last),
        .m_axis_tvalid(m_valid), .m_axis_tready(m_ready), .m_axis_tdata(m_data),
        .m_axis_tkeep(m_keep), .m_axis_tlast(m_last), .interrupt(interrupt)
    );

    always #2.5 clk = ~clk;

    task automatic axi_write(input logic [7:0] address, input logic [31:0] value);
        begin
            @(negedge clk);
            awaddr = address;
            wdata = value;
            awvalid = 1'b1;
            wvalid = 1'b1;
            while (!(awready && wready)) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            awvalid = 1'b0;
            wvalid = 1'b0;
            while (!bvalid) @(negedge clk);
            if (bresp != 0) errors++;
            bready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [7:0] address, output logic [31:0] value);
        begin
            @(negedge clk);
            araddr = address;
            arvalid = 1'b1;
            while (!arready) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            arvalid = 1'b0;
            while (!rvalid) @(negedge clk);
            value = rdata;
            if (rresp != 0) errors++;
            rready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            rready = 1'b0;
        end
    endtask

    task automatic send_word(input logic [63:0] value, input logic last);
        begin
            @(negedge clk);
            s_data = value;
            s_last = last;
            s_valid = 1'b1;
            while (!s_ready) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            s_valid = 1'b0;
            s_last = 1'b0;
        end
    endtask

    initial begin
        logic [31:0] value;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        axi_read(8'h2c, value);
        if (value != 32'h1ff) errors++;
        axi_read(8'h30, value);
        if (value != 32'h00010007) errors++;
        axi_read(8'h94, value);
        if (value != 0) errors++;
        axi_read(8'h98, value);
        if (value != 0) errors++;

        axi_write(8'h90, 32'd0);
        axi_write(8'h00, 32'h00000104);
        send_word({1'b1, 31'd0, 32'd0}, 1'b0);
        send_word(64'h000000000000000f, 1'b0);
        send_word(64'd0, 1'b1);

        wait (interrupt);
        axi_read(8'h04, value);
        if (!value[2] || value[5] || value[1] || value[0]) begin
            errors++;
            $display("ERROR descriptor status=%h", value);
        end
        axi_read(8'h8c, value);
        if (value != 1) begin
            errors++;
            $display("ERROR descriptor records=%0d", value);
        end
        axi_read(8'h0c, value);
        if (value != 1) begin
            errors++;
            $display("ERROR descriptor commands=%0d", value);
        end
        if (errors != 0) begin
            $display("tb_neuraedge_pynq_descriptor FAILED errors=%0d", errors);
        end else begin
            $display("tb_neuraedge_pynq_descriptor PASSED records=1");
        end
        $finish;
    end

    initial begin
        #20_000;
        $fatal(1, "PYNQ descriptor watchdog timeout");
    end
endmodule

`default_nettype wire
