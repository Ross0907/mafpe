`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_pynq_axi_top;
    logic aclk = 1'b0;
    logic aresetn = 1'b0;
    logic [7:0] s_axi_awaddr = 8'd0;
    logic s_axi_awvalid = 1'b0;
    logic s_axi_awready;
    logic [31:0] s_axi_wdata = 32'd0;
    logic [3:0] s_axi_wstrb = 4'd0;
    logic s_axi_wvalid = 1'b0;
    logic s_axi_wready;
    logic [1:0] s_axi_bresp;
    logic s_axi_bvalid;
    logic s_axi_bready = 1'b0;
    logic [7:0] s_axi_araddr = 8'd0;
    logic s_axi_arvalid = 1'b0;
    logic s_axi_arready;
    logic [31:0] s_axi_rdata;
    logic [1:0] s_axi_rresp;
    logic s_axi_rvalid;
    logic s_axi_rready = 1'b0;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic [63:0] s_axis_tdata = 64'd0;
    logic [7:0] s_axis_tkeep = 8'hff;
    logic s_axis_tlast = 1'b0;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b0;
    logic [63:0] m_axis_tdata;
    logic [7:0] m_axis_tkeep;
    logic m_axis_tlast;
    logic interrupt;
    int errors = 0;

    neuraedge_pynq_axi_top dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready), .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready), .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready), .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tlast(s_axis_tlast), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tdata(m_axis_tdata),
        .m_axis_tkeep(m_axis_tkeep), .m_axis_tlast(m_axis_tlast),
        .interrupt(interrupt)
    );

    always #2.5 aclk = ~aclk;

    task automatic send_aw(input logic [7:0] address);
        begin
            @(negedge aclk);
            s_axi_awaddr = address;
            s_axi_awvalid = 1'b1;
            while (!s_axi_awready) @(negedge aclk);
            @(posedge aclk);
            @(negedge aclk);
            s_axi_awvalid = 1'b0;
        end
    endtask

    task automatic send_w(input logic [31:0] value, input logic [3:0] strobes);
        begin
            @(negedge aclk);
            s_axi_wdata = value;
            s_axi_wstrb = strobes;
            s_axi_wvalid = 1'b1;
            while (!s_axi_wready) @(negedge aclk);
            @(posedge aclk);
            @(negedge aclk);
            s_axi_wvalid = 1'b0;
        end
    endtask

    task automatic axi_write(
        input logic [7:0] address,
        input logic [31:0] value,
        input logic [3:0] strobes,
        input logic address_first
    );
        begin
            if (address_first) begin
                send_aw(address);
                repeat (2) @(negedge aclk);
                send_w(value, strobes);
            end else begin
                send_w(value, strobes);
                repeat (3) @(negedge aclk);
                send_aw(address);
            end
            while (!s_axi_bvalid) @(negedge aclk);
            if (s_axi_bresp != 2'b00) begin
                errors++;
                $display("ERROR AXI write response address=%h response=%b", address, s_axi_bresp);
            end
            s_axi_bready = 1'b1;
            @(posedge aclk);
            @(negedge aclk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axi_read(input logic [7:0] address, output logic [31:0] value);
        begin
            @(negedge aclk);
            s_axi_araddr = address;
            s_axi_arvalid = 1'b1;
            while (!s_axi_arready) @(negedge aclk);
            @(posedge aclk);
            @(negedge aclk);
            s_axi_arvalid = 1'b0;
            while (!s_axi_rvalid) @(negedge aclk);
            value = s_axi_rdata;
            if (s_axi_rresp != 2'b00) begin
                errors++;
                $display("ERROR AXI read response address=%h response=%b", address, s_axi_rresp);
            end
            repeat (2) @(negedge aclk);
            if (!s_axi_rvalid || s_axi_rdata !== value) begin
                errors++;
                $display("ERROR AXI read data did not remain stable under backpressure");
            end
            s_axi_rready = 1'b1;
            @(posedge aclk);
            @(negedge aclk);
            s_axi_rready = 1'b0;
        end
    endtask

    task automatic wait_interrupt;
        int timeout;
        begin
            timeout = 0;
            while (!interrupt && timeout < 10000) begin
                @(negedge aclk);
                timeout++;
            end
            if (!interrupt) begin
                errors++;
                $display("ERROR AXI interrupt timeout");
            end
        end
    endtask

    task automatic send_axis_word(
        input logic [63:0] value,
        input logic last_value
    );
        begin
            @(negedge aclk);
            s_axis_tdata = value;
            s_axis_tlast = last_value;
            s_axis_tvalid = 1'b1;
            while (!s_axis_tready) @(negedge aclk);
            @(posedge aclk);
            @(negedge aclk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tdata = 64'd0;
        end
    endtask

    initial begin
        logic [31:0] read_value;
        repeat (5) @(negedge aclk);
        aresetn = 1'b1;

        axi_read(8'h34, read_value);
        if (read_value != 32'h4e455552) begin
            errors++;
            $display("ERROR AXI identity got=%h", read_value);
        end
        axi_read(8'h2c, read_value);
        if (read_value != 32'h1ff) errors++;
        axi_read(8'h30, read_value);
        if (read_value != 32'h00010007) errors++;
        axi_read(8'h28, read_value);
        if (read_value != 32'd200000000) errors++;
        axi_read(8'h48, read_value);
        if (read_value != 0) errors++;
        axi_read(8'h74, read_value);
        if (read_value != 0) errors++;
        axi_read(8'h80, read_value);
        if (read_value != 0) errors++;
        axi_read(8'h94, read_value);
        if (read_value != 0) errors++;
        axi_read(8'h98, read_value);
        if (read_value != 0) errors++;

        // Write address and data channels in both legal arrival orders.
        axi_write(8'h10, 32'h0000000f, 4'hf, 1'b1);
        axi_write(8'h14, 32'h00000000, 4'hf, 1'b0);
        axi_write(8'h18, 32'h00000000, 4'hf, 1'b1);
        axi_write(8'h1c, 32'h00000000, 4'hf, 1'b0);
        axi_write(8'h00, 32'h00000101, 4'hf, 1'b1);
        wait_interrupt();
        axi_read(8'h04, read_value);
        if (!read_value[4] || !read_value[2] || read_value[1] || read_value[0]) begin
            errors++;
            $display("ERROR AXI successful status=%h", read_value);
        end
        axi_read(8'h0c, read_value);
        if (read_value != 1) begin
            errors++;
            $display("ERROR AXI command count=%0d", read_value);
        end

        // Four output words fill the wrapper's two-entry FIFO plus capture
        // register. Hold the external sink stalled long enough to prove that
        // registered credit stops the core without dropping or changing data.
        axi_write(8'h10, 32'h00000009, 4'hf, 1'b1);
        axi_write(8'h14, 32'h00000000, 4'hf, 1'b0);
        axi_write(8'h18, 32'h00000000, 4'hf, 1'b1);
        axi_write(8'h1c, 32'h00000000, 4'hf, 1'b0);
        axi_write(8'h00, 32'h00000101, 4'hf, 1'b1);
        send_axis_word(64'h7fffffff80000000, 1'b0);
        for (int beat = 0; beat < 4; beat++) begin
            send_axis_word({32'(11 + 2*beat), 32'(10 + 2*beat)}, 1'b0);
        end
        for (int beat = 0; beat < 4; beat++) begin
            send_axis_word({32'(3 * (2*beat + 1)), 32'(3 * (2*beat))}, beat == 3);
        end
        while (!m_axis_tvalid) @(negedge aclk);
        begin
            logic [63:0] stalled_data;
            logic stalled_last;
            stalled_data = m_axis_tdata;
            stalled_last = m_axis_tlast;
            repeat (12) begin
                @(negedge aclk);
                if (!m_axis_tvalid || m_axis_tdata !== stalled_data
                        || m_axis_tlast !== stalled_last) begin
                    errors++;
                    $display("ERROR AXI output changed while backpressured");
                end
            end
        end
        for (int beat = 0; beat < 4; beat++) begin
            logic [63:0] expected_word;
            expected_word = {32'(14 + 8*beat), 32'(10 + 8*beat)};
            while (!m_axis_tvalid) @(negedge aclk);
            if (m_axis_tdata !== expected_word || m_axis_tlast != (beat == 3)
                    || m_axis_tkeep != 8'hff) begin
                errors++;
                $display("ERROR AXI vector beat=%0d got=%h last=%0d expected=%h",
                         beat, m_axis_tdata, m_axis_tlast, expected_word);
            end
            m_axis_tready = 1'b1;
            @(posedge aclk);
            @(negedge aclk);
            m_axis_tready = 1'b0;
        end
        wait_interrupt();
        axi_read(8'h20, read_value);
        if (read_value != 9) begin
            errors++;
            $display("ERROR AXI vector input-beat count=%0d", read_value);
        end
        axi_read(8'h24, read_value);
        if (read_value != 4) begin
            errors++;
            $display("ERROR AXI vector output-beat count=%0d", read_value);
        end

        // Byte strobes update only the selected command byte.
        axi_write(8'h10, 32'hdeadbe00, 4'b0001, 1'b0);
        axi_read(8'h10, read_value);
        if (read_value != 32'h00000000) begin
            errors++;
            $display("ERROR AXI write strobe result=%h", read_value);
        end

        // Opcode zero is illegal and must set the core error bit.
        axi_write(8'h00, 32'h00000102, 4'hf, 1'b1);
        axi_write(8'h00, 32'h00000101, 4'hf, 1'b0);
        wait_interrupt();
        axi_read(8'h04, read_value);
        if (!read_value[1] || !read_value[2]) begin
            errors++;
            $display("ERROR AXI invalid-command status=%h", read_value);
        end

        // Restore opcode F and clear all nested sticky errors once more.
        axi_write(8'h10, 32'h0000000f, 4'b0001, 1'b1);
        axi_write(8'h00, 32'h00000102, 4'hf, 1'b0);
        axi_write(8'h00, 32'h00000101, 4'hf, 1'b1);
        wait_interrupt();
        axi_read(8'h04, read_value);
        if (read_value[1] || !read_value[2]) begin
            errors++;
            $display("ERROR AXI final clear status=%h", read_value);
        end
        axi_read(8'h0c, read_value);
        if (read_value != 4) begin
            errors++;
            $display("ERROR AXI final command count=%0d", read_value);
        end

        if (errors != 0) begin
            $display("tb_neuraedge_pynq_axi_top FAILED errors=%0d", errors);
            $finish;
        end else begin
            $display("tb_neuraedge_pynq_axi_top PASSED commands=%0d", read_value);
            $finish;
        end
    end

    initial begin
        #20_000;
        $fatal(1, "tb_neuraedge_pynq_axi_top watchdog timeout");
    end
endmodule

`default_nettype wire
