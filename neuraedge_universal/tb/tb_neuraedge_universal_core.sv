`timescale 1ns / 1ps
`default_nettype none
`include "tb/fixtures/operator_fixture_params.svh"

module tb_neuraedge_universal_core;
    import neuraedge_command_pkg::*;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic command_valid = 1'b0;
    logic command_ready;
    logic [127:0] command_data = 128'd0;
    logic command_done;
    logic status_busy;
    logic status_error;
    logic [3:0] status_active_opcode;
    logic [31:0] status_command_count;
    logic [31:0] status_input_beats;
    logic [31:0] status_output_beats;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic [63:0] s_axis_tdata = 64'd0;
    logic s_axis_tlast = 1'b0;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b0;
    logic [63:0] m_axis_tdata;
    logic m_axis_tlast;

    logic [191:0] vector_command_mem [0:`NEURAEDGE_VECTOR_CASES-1];
    logic [255:0] vector_a_mem [0:`NEURAEDGE_VECTOR_CASES-1];
    logic [255:0] vector_b_mem [0:`NEURAEDGE_VECTOR_CASES-1];
    logic [255:0] vector_expected_mem [0:`NEURAEDGE_VECTOR_CASES-1];
    logic [63:0] optimizer_parameter_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic [127:0] optimizer_weight_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic [191:0] optimizer_gradient_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic [191:0] optimizer_velocity_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic [127:0] optimizer_expected_weight_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic [191:0] optimizer_expected_velocity_mem [0:`NEURAEDGE_OPTIMIZER_CASES-1];
    logic signed [23:0] ln_value_mem [0:`NEURAEDGE_LAYERNORM_SAMPLES-1];
    logic signed [23:0] ln_gamma_mem [0:`NEURAEDGE_LAYERNORM_SAMPLES-1];
    logic signed [23:0] ln_beta_mem [0:`NEURAEDGE_LAYERNORM_SAMPLES-1];
    logic signed [31:0] ln_expected_mem [0:`NEURAEDGE_LAYERNORM_SAMPLES-1];
    logic signed [23:0] sm_score_mem [0:`NEURAEDGE_SOFTMAX_SAMPLES-1];
    logic [15:0] sm_expected_mem [0:`NEURAEDGE_SOFTMAX_SAMPLES-1];
    integer train_expected_dx [0:1][0:7];
    integer train_expected_dw [0:2][0:7];
    logic [31:0] debug_training_tiles;
    logic [31:0] debug_training_cycles;

    int errors = 0;
    int commands_expected = 0;

    neuraedge_universal_core dut (
        .clk(clk), .rst_n(rst_n),
        .command_valid(command_valid), .command_ready(command_ready),
        .command_data(command_data), .command_done(command_done),
        .status_busy(status_busy), .status_error(status_error),
        .status_active_opcode(status_active_opcode),
        .status_command_count(status_command_count),
        .status_input_beats(status_input_beats),
        .status_output_beats(status_output_beats),
        .debug_training_tiles(debug_training_tiles),
        .debug_training_cycles(debug_training_cycles),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata), .m_axis_tlast(m_axis_tlast)
    );

    always #2.5 clk = ~clk;

    task automatic submit_command(input logic [127:0] value);
        begin
            while (!command_ready) @(negedge clk);
            @(negedge clk);
            command_data = value;
            command_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            command_valid = 1'b0;
            command_data = 128'd0;
            commands_expected++;
        end
    endtask

    task automatic send_word(input logic [63:0] value, input logic last_value);
        begin
            while (!s_axis_tready) @(negedge clk);
            s_axis_tdata = value;
            s_axis_tlast = last_value;
            s_axis_tvalid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            s_axis_tvalid = 1'b0;
            s_axis_tlast = 1'b0;
            s_axis_tdata = 64'd0;
        end
    endtask

    task automatic receive_word(
        output logic [63:0] value,
        output logic last_value,
        input int stall_cycles
    );
        begin
            while (!m_axis_tvalid) @(negedge clk);
            repeat (stall_cycles) @(negedge clk);
            value = m_axis_tdata;
            last_value = m_axis_tlast;
            m_axis_tready = 1'b1;
            @(posedge clk);
            @(negedge clk);
            m_axis_tready = 1'b0;
        end
    endtask

    task automatic wait_done;
        int timeout;
        begin
            timeout = 0;
            while (!command_done && timeout < 100000) begin
                @(negedge clk);
                timeout++;
            end
            if (!command_done) begin
                errors++;
                $display("ERROR universal command timeout opcode=%0d state=%0d",
                         status_active_opcode, dut.state);
            end
        end
    endtask

    function automatic logic [127:0] tensor_command(
        input logic [3:0] opcode,
        input int rows,
        input int columns,
        input int count,
        input int activation_zp,
        input int weight_zp
    );
        logic [127:0] value;
        begin
            value = 128'd0;
            value[3:0] = opcode;
            value[7:4] = 4'(rows);
            value[13:8] = 6'(columns);
            value[25:14] = 12'(count);
            value[33:26] = 8'(activation_zp);
            value[41:34] = 8'(weight_zp);
            tensor_command = value;
        end
    endfunction

    initial begin
        logic [127:0] command;
        logic [63:0] received_word;
        logic received_last;
        logic [255:0] received_vector;
        logic [127:0] received_weights;
        logic [191:0] received_velocities;
        logic [767:0] received_lif_trace;
        logic [511:0] received_train_dx;
        logic [575:0] received_train_dw;
        logic [255:0] alpha_query;
        logic [255:0] alpha_key;
        int alpha_spikes;
        int alpha_zeros;
        int alpha_expected;
        int seed;

        $readmemh("tb/fixtures/vector_commands.mem", vector_command_mem);
        $readmemh("tb/fixtures/vector_a.mem", vector_a_mem);
        $readmemh("tb/fixtures/vector_b.mem", vector_b_mem);
        $readmemh("tb/fixtures/vector_expected.mem", vector_expected_mem);
        $readmemh("tb/fixtures/optimizer_parameters.mem", optimizer_parameter_mem);
        $readmemh("tb/fixtures/optimizer_weights.mem", optimizer_weight_mem);
        $readmemh("tb/fixtures/optimizer_gradients.mem", optimizer_gradient_mem);
        $readmemh("tb/fixtures/optimizer_velocities.mem", optimizer_velocity_mem);
        $readmemh("tb/fixtures/optimizer_expected_weights.mem", optimizer_expected_weight_mem);
        $readmemh("tb/fixtures/optimizer_expected_velocities.mem", optimizer_expected_velocity_mem);
        $readmemh("tb/fixtures/layernorm_values.mem", ln_value_mem);
        $readmemh("tb/fixtures/layernorm_gammas.mem", ln_gamma_mem);
        $readmemh("tb/fixtures/layernorm_betas.mem", ln_beta_mem);
        $readmemh("tb/fixtures/layernorm_expected.mem", ln_expected_mem);
        $readmemh("tb/fixtures/softmax_scores.mem", sm_score_mem);
        $readmemh("tb/fixtures/softmax_expected.mem", sm_expected_mem);

        seed = 32'hc01ec7ed;
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        // Clear every sticky status source, including the nested tensor core.
        submit_command({124'd0, CMD_CLEAR_STATUS});
        wait_done();

        // Empty centered tensor tile: compaction emits one sentinel issue and
        // the exact corrected result remains zero.
        submit_command(tensor_command(CMD_LOAD_ACTIVATIONS, 0, 0, 1, 128, 0));
        send_word(64'h8080808080808080, 1'b1);
        wait_done();
        submit_command(tensor_command(CMD_LOAD_WEIGHTS_OUTER, 0, 0, 1, 0, 0));
        send_word(64'h0102030405060708, 1'b0);
        send_word(64'h1112131415161718, 1'b1);
        wait_done();
        submit_command(tensor_command(CMD_COMPUTE_DENSE_OUTER, 1, 1, 1, 128, 0));
        receive_word(received_word, received_last, 2);
        if (!received_last || $signed(received_word[47:0]) != 0) begin
            errors++;
            $display("ERROR universal tensor result=%0d last=%0d", $signed(received_word[47:0]), received_last);
        end

        // Alpha-XNOR command and eight-beat binary vector transfer.
        for (int word_index = 0; word_index < 8; word_index++) begin
            alpha_query[word_index*32 +: 32] = $urandom(seed);
            alpha_key[word_index*32 +: 32] = $urandom(seed);
        end
        alpha_spikes = 0;
        alpha_zeros = 0;
        for (int bit_index = 0; bit_index < 256; bit_index++) begin
            alpha_spikes += alpha_query[bit_index] && alpha_key[bit_index];
            alpha_zeros += !alpha_query[bit_index] && !alpha_key[bit_index];
        end
        alpha_expected = alpha_spikes * 256 + alpha_zeros * 127;
        command = 128'd0;
        command[3:0] = CMD_ALPHA_XNOR;
        command[50:42] = 9'd256;
        command[59:51] = 9'd127;
        submit_command(command);
        for (int beat = 0; beat < 4; beat++) send_word(alpha_query[beat*64 +: 64], 1'b0);
        for (int beat = 0; beat < 4; beat++) send_word(alpha_key[beat*64 +: 64], beat == 3);
        receive_word(received_word, received_last, 3);
        if (!received_last || received_word[17:0] != alpha_expected
                || received_word[26:18] != alpha_spikes || received_word[35:27] != alpha_zeros) begin
            errors++;
            $display("ERROR universal alpha-XNOR result");
        end

        // Vector command uses fixture case 50 (REQUANTIZE) through the shared stream.
        command = 128'd0;
        command[3:0] = CMD_VECTOR_ALU;
        command[108:4] = vector_command_mem[50][104:0];
        submit_command(command);
        send_word({vector_command_mem[50][168:137], vector_command_mem[50][136:105]}, 1'b0);
        for (int beat = 0; beat < 4; beat++) send_word(vector_a_mem[50][beat*64 +: 64], 1'b0);
        for (int beat = 0; beat < 4; beat++) send_word(vector_b_mem[50][beat*64 +: 64], beat == 3);
        for (int beat = 0; beat < 4; beat++) begin
            receive_word(received_word, received_last, beat % 2);
            received_vector[beat*64 +: 64] = received_word;
            if (received_last != (beat == 3)) errors++;
        end
        if (received_vector !== vector_expected_mem[50]) begin
            errors++;
            $display("ERROR universal vector result");
        end

        // LayerNorm fixture case 2 starts after the first five samples.
        command = 128'd0;
        command[3:0] = CMD_LAYERNORM;
        command[15:4] = 12'd7;
        command[47:16] = 32'd7;
        submit_command(command);
        for (int sample = 0; sample < 7; sample++) begin
            logic [63:0] first_word;
            first_word = {ln_beta_mem[5+sample][15:0], ln_gamma_mem[5+sample], ln_value_mem[5+sample]};
            send_word(first_word, 1'b0);
            send_word({56'd0, ln_beta_mem[5+sample][23:16]}, sample == 6);
        end
        for (int sample = 0; sample < 7; sample++) begin
            receive_word(received_word, received_last, sample % 2);
            if ($signed(received_word[31:0]) !== ln_expected_mem[5+sample]
                    || received_last != (sample == 6)) begin
                errors++;
                $display("ERROR universal LayerNorm sample=%0d got=%0d expected=%0d",
                         sample, $signed(received_word[31:0]), ln_expected_mem[5+sample]);
            end
        end

        // Softmax fixture case 2 also starts after five entries.
        command = 128'd0;
        command[3:0] = CMD_LOG_SOFTMAX;
        command[12:4] = 9'd7;
        command[17:13] = 5'd2;
        submit_command(command);
        for (int sample = 0; sample < 7; sample++) begin
            send_word({40'd0, sm_score_mem[5+sample]}, sample == 6);
        end
        for (int sample = 0; sample < 7; sample++) begin
            receive_word(received_word, received_last, 1);
            if (received_word[15:0] !== sm_expected_mem[5+sample]
                    || received_last != (sample == 6)) begin
                errors++;
                $display("ERROR universal softmax sample=%0d", sample);
            end
        end

        // One state-clearing LIF timestep with analytically obvious currents.
        command = 128'd0;
        command[3:0] = CMD_LIF_STEP;
        command[8:4] = 5'd2;
        command[32:9] = 24'sd50;
        command[56:33] = 24'sd0;
        command[57] = 1'b1;
        submit_command(command);
        for (int beat = 0; beat < 8; beat++) begin
            logic [63:0] currents;
            currents = 64'd0;
            for (int lane = 0; lane < 4; lane++) begin
                currents[lane*16 +: 16] = 16'(5 * (beat*4 + lane));
            end
            send_word(currents, beat == 7);
        end
        receive_word(received_word, received_last, 2);
        if (!received_last || received_word[63:32] != 1
                || received_word[31:0] != 32'hfffffc00) begin
            errors++;
            $display("ERROR universal LIF got=%h", received_word);
        end

        // Trace mode returns the same metadata plus every pre-reset membrane.
        // These are the h[t] values required by surrogate-gradient BPTT.
        command[58] = 1'b1;
        submit_command(command);
        for (int beat = 0; beat < 8; beat++) begin
            logic [63:0] currents;
            currents = 64'd0;
            for (int lane = 0; lane < 4; lane++) begin
                currents[lane*16 +: 16] = 16'(5 * (beat*4 + lane));
            end
            send_word(currents, beat == 7);
        end
        receive_word(received_word, received_last, 1);
        if (received_last || received_word[63:32] != 1
                || received_word[31:0] != 32'hfffffc00) begin
            errors++;
            $display("ERROR universal traced LIF metadata got=%h last=%b",
                     received_word, received_last);
        end
        received_lif_trace = '0;
        for (int beat = 0; beat < 12; beat++) begin
            receive_word(received_word, received_last, beat % 3);
            received_lif_trace[beat*64 +: 64] = received_word;
            if (received_last != (beat == 11)) errors++;
        end
        for (int lane = 0; lane < 32; lane++) begin
            if ($signed(received_lif_trace[lane*24 +: 24]) != 5 * lane) begin
                errors++;
                if (errors <= 20) begin
                    $display("ERROR universal traced LIF lane=%0d got=%0d expected=%0d",
                             lane,
                             $signed(received_lif_trace[lane*24 +: 24]),
                             5 * lane);
                end
            end
        end

        // Full eight-lane on-chip SGDW update using fixture case zero.
        command = 128'd0;
        command[3:0] = CMD_SGD_UPDATE;
        command[21:4] = optimizer_parameter_mem[0][17:0];
        command[38:22] = optimizer_parameter_mem[0][34:18];
        command[55:39] = optimizer_parameter_mem[0][51:35];
        command[60:56] = optimizer_parameter_mem[0][56:52];
        submit_command(command);
        for (int lane = 0; lane < 8; lane++) begin
            send_word({optimizer_velocity_mem[0][lane*24 +: 24],
                       optimizer_gradient_mem[0][lane*24 +: 24],
                       optimizer_weight_mem[0][lane*16 +: 16]}, lane == 7);
        end
        for (int lane = 0; lane < 8; lane++) begin
            receive_word(received_word, received_last, lane % 3);
            received_weights[lane*16 +: 16] = received_word[15:0];
            received_velocities[lane*24 +: 24] = received_word[39:16];
            if (received_last != (lane == 7)) errors++;
        end
        if (received_weights !== optimizer_expected_weight_mem[0]
                || received_velocities !== optimizer_expected_velocity_mem[0]) begin
            errors++;
            $display("ERROR universal optimizer result");
        end

        // Backpropagate one complete Y=XW tile through the universal stream.
        // X, W, and dY are s16; dX is shifted once and dW is unshifted.
        command = 128'd0;
        command[3:0] = CMD_TRAIN_LINEAR_BACKWARD;
        command[7:4] = 4'd2;
        command[11:8] = 4'd3;
        command[15:12] = 4'd2;
        command[20:16] = 5'd1;
        command[25:21] = 5'd0;
        submit_command(command);
        send_word(64'h00000003fffe0001, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h0000fffa0005fffc, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h00000000fff80007, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h00000000000a0009, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h00000000000cfff5, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h00000000fff2000d, 1'b0);
        send_word(64'h0000000000000000, 1'b0);
        send_word(64'h000000000010fff1, 1'b0);
        send_word(64'h0000000000000000, 1'b1);
        received_train_dx = '0;
        for (int beat = 0; beat < 8; beat++) begin
            receive_word(received_word, received_last, beat % 3);
            received_train_dx[beat*64 +: 64] = received_word;
            if (received_last) errors++;
        end
        received_train_dw = '0;
        for (int beat = 0; beat < 9; beat++) begin
            receive_word(received_word, received_last, (beat + 1) % 3);
            received_train_dw[beat*64 +: 64] = received_word;
            if (received_last != (beat == 8)) errors++;
        end
        train_expected_dx[0][0] = 102;
        train_expected_dx[0][1] = -12;
        train_expected_dx[0][2] = -156;
        train_expected_dx[1][0] = -117;
        train_expected_dx[1][1] = 13;
        train_expected_dx[1][2] = 179;
        for (int row = 0; row < 2; row++) begin
            for (int lane = 3; lane < 8; lane++) train_expected_dx[row][lane] = 0;
            for (int lane = 0; lane < 8; lane++) begin
                if ($signed(received_train_dx[(row*8+lane)*32 +: 32])
                        != train_expected_dx[row][lane]) begin
                    errors++;
                    $display("ERROR universal backward dX row=%0d lane=%0d got=%0d expected=%0d",
                             row, lane,
                             $signed(received_train_dx[(row*8+lane)*32 +: 32]),
                             train_expected_dx[row][lane]);
                end
            end
        end
        train_expected_dw[0][0] = 73;
        train_expected_dw[0][1] = -78;
        train_expected_dw[1][0] = -101;
        train_expected_dw[1][1] = 108;
        train_expected_dw[2][0] = 129;
        train_expected_dw[2][1] = -138;
        for (int row = 0; row < 3; row++) begin
            for (int lane = 2; lane < 8; lane++) train_expected_dw[row][lane] = 0;
            for (int lane = 0; lane < 8; lane++) begin
                if ($signed(received_train_dw[row*192+lane*24 +: 24])
                        != train_expected_dw[row][lane]) begin
                    errors++;
                    $display("ERROR universal backward dW row=%0d lane=%0d got=%0d expected=%0d",
                             row, lane,
                             $signed(received_train_dw[row*192+lane*24 +: 24]),
                             train_expected_dw[row][lane]);
                end
            end
        end
        wait_done();

        // Program and evaluate all eight ports of the activation LUT.
        command = 128'd0;
        command[3:0] = CMD_ACTIVATION_LUT;
        command[7:4] = 4'd0;
        submit_command(command);
        wait_done();
        command[7:4] = 4'd1;
        command[19:8] = 12'd8;
        submit_command(command);
        for (int address = 0; address < 8; address++) begin
            send_word({40'd0, 16'(address*address-3), 8'(address)}, address == 7);
        end
        wait_done();
        command = 128'd0;
        command[3:0] = CMD_ACTIVATION_LUT;
        command[7:4] = 4'd2;
        submit_command(command);
        send_word(64'h0706050403020100, 1'b1);
        received_vector = '0;
        for (int beat = 0; beat < 2; beat++) begin
            receive_word(received_word, received_last, beat);
            received_vector[beat*64 +: 64] = received_word;
        end
        for (int lane = 0; lane < 8; lane++) begin
            if ($signed(received_vector[lane*16 +: 16]) != lane*lane-3) begin
                errors++;
                $display("ERROR universal LUT lane=%0d", lane);
            end
        end
        wait_done();

        if (status_error || status_busy || status_command_count != commands_expected
                || status_input_beats != 88 || status_output_beats != 61
                || debug_training_tiles != 1 || debug_training_cycles == 0) begin
            errors++;
            $display("ERROR universal final status error=%0d busy=%0d commands=%0d/%0d inputs=%0d outputs=%0d training=%0d/%0d",
                     status_error, status_busy, status_command_count, commands_expected,
                     status_input_beats, status_output_beats,
                     debug_training_tiles, debug_training_cycles);
        end
        if (errors != 0) begin
            $display("tb_neuraedge_universal_core FAILED errors=%0d", errors);
            $finish;
        end else begin
            $display("tb_neuraedge_universal_core PASSED commands=%0d input_beats=%0d output_beats=%0d",
                     status_command_count, status_input_beats, status_output_beats);
            $finish;
        end
    end

    initial begin
        #2_000_000;
        $fatal(1, "tb_neuraedge_universal_core watchdog timeout state=%0d opcode=%0d",
               dut.state, status_active_opcode);
    end
endmodule

`default_nettype wire
