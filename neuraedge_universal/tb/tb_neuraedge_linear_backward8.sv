`timescale 1ns / 1ps
`default_nettype none

module tb_neuraedge_linear_backward8;
    localparam int LANES = 8;
    localparam int RANDOM_CASES = 48;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic clear_counters = 1'b0;
    logic start_valid = 1'b0;
    logic start_ready;
    logic [3:0] active_rows = 4'd0;
    logic [3:0] active_inputs = 4'd0;
    logic [3:0] active_outputs = 4'd0;
    logic [4:0] input_gradient_shift = 5'd0;
    logic [4:0] weight_gradient_shift = 5'd0;
    logic [1:0] gradient_mode = 2'd0;
    logic accumulate_result = 1'b0;
    logic suppress_output = 1'b0;
    logic reserved_nonzero = 1'b0;
    logic s_axis_tvalid = 1'b0;
    logic s_axis_tready;
    logic [63:0] s_axis_tdata = 64'd0;
    logic s_axis_tlast = 1'b0;
    logic m_axis_tvalid;
    logic m_axis_tready = 1'b0;
    logic [63:0] m_axis_tdata;
    logic m_axis_tlast;
    logic busy;
    logic done;
    logic config_error;
    logic [31:0] completed_tiles;
    logic [31:0] active_cycles;

    integer activation [0:LANES-1][0:LANES-1];
    integer weight [0:LANES-1][0:LANES-1];
    integer output_gradient [0:LANES-1][0:LANES-1];
    logic signed [31:0] expected_input_gradient [0:LANES-1][0:LANES-1];
    logic signed [23:0] expected_weight_gradient [0:LANES-1][0:LANES-1];

    integer errors = 0;
    integer checked_tiles = 0;
    integer seed;
    logic held_valid = 1'b0;
    logic [63:0] held_data;
    logic held_last;

    neuraedge_linear_backward8 dut (
        .clk(clk),
        .rst_n(rst_n),
        .clear_counters(clear_counters),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .active_rows(active_rows),
        .active_inputs(active_inputs),
        .active_outputs(active_outputs),
        .input_gradient_shift(input_gradient_shift),
        .weight_gradient_shift(weight_gradient_shift),
        .gradient_mode(gradient_mode),
        .accumulate_result(accumulate_result),
        .suppress_output(suppress_output),
        .reserved_nonzero(reserved_nonzero),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata(s_axis_tdata),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tlast(m_axis_tlast),
        .busy(busy),
        .done(done),
        .config_error(config_error),
        .completed_tiles(completed_tiles),
        .active_cycles(active_cycles)
    );

    always #2.5 clk = ~clk;

    always @(posedge clk) begin
        if (held_valid && (!m_axis_tvalid || m_axis_tdata !== held_data
                || m_axis_tlast !== held_last)) begin
            errors = errors + 1;
            $display("ERROR backward output changed while stalled");
        end
        held_valid <= m_axis_tvalid && !m_axis_tready;
        held_data <= m_axis_tdata;
        held_last <= m_axis_tlast;
    end

    function automatic signed [63:0] round_shift_ref(
        input signed [63:0] value,
        input integer shift
    );
        reg signed [63:0] magnitude;
        reg signed [63:0] rounded;
        begin
            if (shift == 0) begin
                round_shift_ref = value;
            end else begin
                magnitude = value < 0 ? -value : value;
                rounded = (magnitude + (64'sd1 << (shift - 1))) >>> shift;
                round_shift_ref = value < 0 ? -rounded : rounded;
            end
        end
    endfunction

    function automatic signed [31:0] saturate_s32_ref(
        input signed [63:0] value
    );
        begin
            if (value > 64'sd2147483647) saturate_s32_ref = 32'sh7fffffff;
            else if (value < -64'sd2147483648) saturate_s32_ref = 32'sh80000000;
            else saturate_s32_ref = value[31:0];
        end
    endfunction

    function automatic signed [23:0] saturate_s24_ref(
        input signed [63:0] value
    );
        begin
            if (value > 64'sd8388607) saturate_s24_ref = 24'sh7fffff;
            else if (value < -64'sd8388608) saturate_s24_ref = 24'sh800000;
            else saturate_s24_ref = value[23:0];
        end
    endfunction

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
        input integer stall_cycles
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

    task automatic start_tile(
        input integer rows,
        input integer inputs,
        input integer outputs,
        input integer dx_shift,
        input integer dw_shift,
        input logic bad_reserved
    );
        begin
            while (!start_ready) @(negedge clk);
            active_rows = rows;
            active_inputs = inputs;
            active_outputs = outputs;
            input_gradient_shift = dx_shift;
            weight_gradient_shift = dw_shift;
            gradient_mode = 2'd0;
            accumulate_result = 1'b0;
            suppress_output = 1'b0;
            reserved_nonzero = bad_reserved;
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
            reserved_nonzero = 1'b0;
        end
    endtask

    task automatic start_partial_tile(
        input integer rows,
        input integer inputs,
        input integer outputs,
        input integer dx_shift,
        input integer dw_shift,
        input logic [1:0] mode_value,
        input logic accumulate_value,
        input logic suppress_value
    );
        begin
            while (!start_ready) @(negedge clk);
            active_rows = rows;
            active_inputs = inputs;
            active_outputs = outputs;
            input_gradient_shift = dx_shift;
            weight_gradient_shift = dw_shift;
            gradient_mode = mode_value;
            accumulate_result = accumulate_value;
            suppress_output = suppress_value;
            reserved_nonzero = 1'b0;
            start_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            start_valid = 1'b0;
        end
    endtask

    task automatic wait_suppressed_done;
        integer timeout;
        begin
            timeout = 0;
            while (!done && timeout < 10000) begin
                @(negedge clk);
                if (m_axis_tvalid) begin
                    errors = errors + 1;
                    $display("ERROR suppressed backward command emitted output");
                end
                timeout = timeout + 1;
            end
            if (!done) begin
                errors = errors + 1;
                $display("ERROR suppressed backward command timed out");
            end
            if (config_error) begin
                errors = errors + 1;
                $display("ERROR suppressed backward command reported configuration failure");
            end
            checked_tiles = checked_tiles + 1;
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    task automatic run_exact_accumulation_modes;
        integer row;
        integer input_index;
        integer output_index;
        integer chunk;
        integer chunk_size;
        integer global_index;
        integer half;
        integer lane;
        integer beat_index;
        integer word_index;
        integer timeout;
        reg signed [63:0] accumulator;
        logic [63:0] stream_word;
        logic [63:0] received_word;
        logic received_last;
        logic [191:0] packed_weight_row;
        begin
            // dX is a reduction over output features. Two commands cover
            // thirteen features while one 47-bit accumulator preserves the
            // exact sum until the final round-by-16 operation.
            for (row = 0; row < LANES; row = row + 1) begin
                for (input_index = 0; input_index < LANES;
                        input_index = input_index + 1) begin
                    expected_input_gradient[row][input_index] = 32'sd0;
                end
            end
            for (row = 0; row < 5; row = row + 1) begin
                for (input_index = 0; input_index < 6;
                        input_index = input_index + 1) begin
                    accumulator = 64'sd0;
                    for (output_index = 0; output_index < 13;
                            output_index = output_index + 1) begin
                        accumulator = accumulator
                            + ((((row*71 + output_index*43 + 11) % 1801) - 900)
                            * (((input_index*97 + output_index*53 + 17) % 2001) - 1000));
                    end
                    expected_input_gradient[row][input_index] = saturate_s32_ref(
                        round_shift_ref(accumulator, 4));
                end
            end
            for (chunk = 0; chunk < 2; chunk = chunk + 1) begin
                chunk_size = chunk == 0 ? 8 : 5;
                start_partial_tile(5, 6, chunk_size, 4, 0, 2'd1,
                                   chunk != 0, chunk == 0);
                for (input_index = 0; input_index < 6;
                        input_index = input_index + 1) begin
                    for (half = 0; half < 2; half = half + 1) begin
                        stream_word = 64'd0;
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            output_index = half * 4 + lane;
                            if (output_index < chunk_size) begin
                                global_index = chunk * 8 + output_index;
                                stream_word[lane*16 +: 16] =
                                    ((input_index*97 + global_index*53 + 17) % 2001) - 1000;
                            end
                        end
                        send_word(stream_word, 1'b0);
                    end
                end
                for (row = 0; row < 5; row = row + 1) begin
                    for (half = 0; half < 2; half = half + 1) begin
                        stream_word = 64'd0;
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            output_index = half * 4 + lane;
                            if (output_index < chunk_size) begin
                                global_index = chunk * 8 + output_index;
                                stream_word[lane*16 +: 16] =
                                    ((row*71 + global_index*43 + 11) % 1801) - 900;
                            end
                        end
                        send_word(stream_word, row == 4 && half == 1);
                    end
                end
                if (chunk == 0) begin
                    wait_suppressed_done();
                end else begin
                    for (beat_index = 0; beat_index < 20;
                            beat_index = beat_index + 1) begin
                        receive_word(received_word, received_last, beat_index % 3);
                        row = beat_index / 4;
                        word_index = beat_index % 4;
                        for (lane = 0; lane < 2; lane = lane + 1) begin
                            input_index = word_index * 2 + lane;
                            if ($signed(received_word[lane*32 +: 32])
                                    !== expected_input_gradient[row][input_index]) begin
                                errors = errors + 1;
                                $display("ERROR accumulated dX row=%0d lane=%0d got=%0d expected=%0d",
                                         row, input_index,
                                         $signed(received_word[lane*32 +: 32]),
                                         expected_input_gradient[row][input_index]);
                            end
                        end
                        if (received_last != (beat_index == 19)) errors = errors + 1;
                    end
                    timeout = 0;
                    while (!done && timeout < 20) begin
                        @(negedge clk);
                        timeout = timeout + 1;
                    end
                    if (!done || config_error) errors = errors + 1;
                    checked_tiles = checked_tiles + 1;
                    @(posedge clk);
                    @(negedge clk);
                end
            end

            // dW is a reduction over rows. Thirteen rows are split 8+5; the
            // dW-only mode consequently omits every weight input beat.
            for (input_index = 0; input_index < LANES;
                    input_index = input_index + 1) begin
                for (output_index = 0; output_index < LANES;
                        output_index = output_index + 1) begin
                    expected_weight_gradient[input_index][output_index] = 24'sd0;
                end
            end
            for (input_index = 0; input_index < 6;
                    input_index = input_index + 1) begin
                for (output_index = 0; output_index < 7;
                        output_index = output_index + 1) begin
                    accumulator = 64'sd0;
                    for (row = 0; row < 13; row = row + 1) begin
                        accumulator = accumulator
                            + ((((row*29 + input_index*13 + 7) % 255) - 127)
                            * (((row*71 + output_index*43 + 11) % 1801) - 900));
                    end
                    expected_weight_gradient[input_index][output_index]
                        = saturate_s24_ref(round_shift_ref(accumulator, 3));
                end
            end
            for (chunk = 0; chunk < 2; chunk = chunk + 1) begin
                chunk_size = chunk == 0 ? 8 : 5;
                start_partial_tile(chunk_size, 6, 7, 0, 3, 2'd2,
                                   chunk != 0, chunk == 0);
                for (row = 0; row < chunk_size; row = row + 1) begin
                    global_index = chunk * 8 + row;
                    for (half = 0; half < 2; half = half + 1) begin
                        stream_word = 64'd0;
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            input_index = half * 4 + lane;
                            if (input_index < 6) begin
                                stream_word[lane*16 +: 16] =
                                    ((global_index*29 + input_index*13 + 7) % 255) - 127;
                            end
                        end
                        send_word(stream_word, 1'b0);
                    end
                end
                for (row = 0; row < chunk_size; row = row + 1) begin
                    global_index = chunk * 8 + row;
                    for (half = 0; half < 2; half = half + 1) begin
                        stream_word = 64'd0;
                        for (lane = 0; lane < 4; lane = lane + 1) begin
                            output_index = half * 4 + lane;
                            if (output_index < 7) begin
                                stream_word[lane*16 +: 16] =
                                    ((global_index*71 + output_index*43 + 11) % 1801) - 900;
                            end
                        end
                        send_word(stream_word,
                                  row == chunk_size - 1 && half == 1);
                    end
                end
                if (chunk == 0) begin
                    wait_suppressed_done();
                end else begin
                    packed_weight_row = 192'd0;
                    for (beat_index = 0; beat_index < 18;
                            beat_index = beat_index + 1) begin
                        receive_word(received_word, received_last, beat_index % 4);
                        input_index = beat_index / 3;
                        word_index = beat_index % 3;
                        packed_weight_row[word_index*64 +: 64] = received_word;
                        if (word_index == 2) begin
                            for (lane = 0; lane < LANES; lane = lane + 1) begin
                                if ($signed(packed_weight_row[lane*24 +: 24])
                                        !== expected_weight_gradient[input_index][lane]) begin
                                    errors = errors + 1;
                                    $display("ERROR accumulated dW row=%0d lane=%0d got=%0d expected=%0d",
                                             input_index, lane,
                                             $signed(packed_weight_row[lane*24 +: 24]),
                                             expected_weight_gradient[input_index][lane]);
                                end
                            end
                            packed_weight_row = 192'd0;
                        end
                        if (received_last != (beat_index == 17)) errors = errors + 1;
                    end
                    timeout = 0;
                    while (!done && timeout < 20) begin
                        @(negedge clk);
                        timeout = timeout + 1;
                    end
                    if (!done || config_error) errors = errors + 1;
                    checked_tiles = checked_tiles + 1;
                    @(posedge clk);
                    @(negedge clk);
                end
            end

            // Emitting the final result invalidates the retained shape.
            start_partial_tile(5, 6, 5, 4, 0, 2'd1, 1'b1, 1'b1);
            while (!done) @(negedge clk);
            if (!config_error) begin
                errors = errors + 1;
                $display("ERROR stale dX accumulator remained valid after final output");
            end
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    task automatic prepare_case(
        input integer rows,
        input integer inputs,
        input integer outputs,
        input integer pattern,
        input integer dx_shift,
        input integer dw_shift
    );
        integer row;
        integer column;
        integer reduction;
        reg signed [63:0] accumulator;
        begin
            for (row = 0; row < LANES; row = row + 1) begin
                for (column = 0; column < LANES; column = column + 1) begin
                    activation[row][column] = 0;
                    weight[row][column] = 0;
                    output_gradient[row][column] = 0;
                    expected_input_gradient[row][column] = 32'sd0;
                    expected_weight_gradient[row][column] = 24'sd0;
                end
            end

            for (row = 0; row < rows; row = row + 1) begin
                for (column = 0; column < inputs; column = column + 1) begin
                    if (pattern == 1) activation[row][column] = 127;
                    else if (pattern == 2) activation[row][column] = -128;
                    else if (pattern == 3) activation[row][column] = 127;
                    else activation[row][column] = $urandom_range(255, 0) - 128;
                end
            end
            for (row = 0; row < inputs; row = row + 1) begin
                for (column = 0; column < outputs; column = column + 1) begin
                    if (pattern == 1) weight[row][column] = -32768;
                    else if (pattern == 2) weight[row][column] = 32767;
                    else if (pattern == 3) weight[row][column] = 32767;
                    else weight[row][column] = $urandom_range(65535, 0) - 32768;
                end
            end
            for (row = 0; row < rows; row = row + 1) begin
                for (column = 0; column < outputs; column = column + 1) begin
                    if (pattern == 1) output_gradient[row][column] = -32768;
                    else if (pattern == 2) output_gradient[row][column] = 32767;
                    else if (pattern == 3) output_gradient[row][column] = -32768;
                    else output_gradient[row][column] = $urandom_range(65535, 0) - 32768;
                end
            end

            for (row = 0; row < rows; row = row + 1) begin
                for (column = 0; column < inputs; column = column + 1) begin
                    accumulator = 64'sd0;
                    for (reduction = 0; reduction < outputs; reduction = reduction + 1) begin
                        accumulator = accumulator
                            + output_gradient[row][reduction] * weight[column][reduction];
                    end
                    expected_input_gradient[row][column] = saturate_s32_ref(
                        round_shift_ref(accumulator, dx_shift));
                end
            end
            for (row = 0; row < inputs; row = row + 1) begin
                for (column = 0; column < outputs; column = column + 1) begin
                    accumulator = 64'sd0;
                    for (reduction = 0; reduction < rows; reduction = reduction + 1) begin
                        accumulator = accumulator
                            + activation[reduction][row] * output_gradient[reduction][column];
                    end
                    expected_weight_gradient[row][column] = saturate_s24_ref(
                        round_shift_ref(accumulator, dw_shift));
                end
            end
        end
    endtask

    task automatic run_case(
        input integer rows,
        input integer inputs,
        input integer outputs,
        input integer pattern,
        input integer dx_shift,
        input integer dw_shift,
        input logic malformed_tlast
    );
        integer row;
        integer lane;
        integer half;
        integer beat_index;
        integer row_index;
        integer word_index;
        integer lane_index;
        integer total_output_beats;
        integer timeout;
        logic [63:0] stream_word;
        logic [63:0] received_word;
        logic received_last;
        logic [191:0] packed_weight_row;
        begin
            prepare_case(rows, inputs, outputs, pattern, dx_shift, dw_shift);
            start_tile(rows, inputs, outputs, dx_shift, dw_shift, 1'b0);

            for (row = 0; row < rows; row = row + 1) begin
                for (half = 0; half < 2; half = half + 1) begin
                    stream_word = 64'd0;
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        stream_word[lane*16 +: 16] = activation[row][half*4 + lane];
                    end
                    send_word(stream_word,
                              malformed_tlast && row == 0 && half == 0);
                end
            end
            for (row = 0; row < inputs; row = row + 1) begin
                for (half = 0; half < 2; half = half + 1) begin
                    stream_word = 64'd0;
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        stream_word[lane*16 +: 16] = weight[row][half*4 + lane];
                    end
                    send_word(stream_word, 1'b0);
                end
            end
            for (row = 0; row < rows; row = row + 1) begin
                for (half = 0; half < 2; half = half + 1) begin
                    stream_word = 64'd0;
                    for (lane = 0; lane < 4; lane = lane + 1) begin
                        stream_word[lane*16 +: 16] = output_gradient[row][half*4 + lane];
                    end
                    send_word(stream_word, row == rows - 1 && half == 1);
                end
            end

            total_output_beats = 4 * rows + 3 * inputs;
            packed_weight_row = 192'd0;
            for (beat_index = 0; beat_index < total_output_beats;
                    beat_index = beat_index + 1) begin
                receive_word(received_word, received_last, beat_index % 4);
                if (received_last != (beat_index == total_output_beats - 1)) begin
                    errors = errors + 1;
                    $display("ERROR backward TLAST case=%0d beat=%0d got=%0b",
                             checked_tiles, beat_index, received_last);
                end
                if (beat_index < 4 * rows) begin
                    row_index = beat_index / 4;
                    word_index = beat_index % 4;
                    for (lane = 0; lane < 2; lane = lane + 1) begin
                        lane_index = word_index * 2 + lane;
                        if ($signed(received_word[lane*32 +: 32])
                                !== expected_input_gradient[row_index][lane_index]) begin
                            errors = errors + 1;
                            if (errors <= 30) begin
                                $display("ERROR dX tile=%0d row=%0d lane=%0d got=%0d expected=%0d",
                                         checked_tiles, row_index, lane_index,
                                         $signed(received_word[lane*32 +: 32]),
                                         expected_input_gradient[row_index][lane_index]);
                            end
                        end
                    end
                end else begin
                    row_index = (beat_index - 4 * rows) / 3;
                    word_index = (beat_index - 4 * rows) % 3;
                    packed_weight_row[word_index*64 +: 64] = received_word;
                    if (word_index == 2) begin
                        for (lane = 0; lane < LANES; lane = lane + 1) begin
                            if ($signed(packed_weight_row[lane*24 +: 24])
                                    !== expected_weight_gradient[row_index][lane]) begin
                                errors = errors + 1;
                                if (errors <= 30) begin
                                    $display("ERROR dW tile=%0d row=%0d lane=%0d got=%0d expected=%0d",
                                             checked_tiles, row_index, lane,
                                             $signed(packed_weight_row[lane*24 +: 24]),
                                             expected_weight_gradient[row_index][lane]);
                                end
                            end
                        end
                        packed_weight_row = 192'd0;
                    end
                end
            end

            timeout = 0;
            while (!done && timeout < 20) begin
                @(negedge clk);
                timeout = timeout + 1;
            end
            if (!done) begin
                errors = errors + 1;
                $display("ERROR backward done timeout");
            end
            if (config_error !== malformed_tlast) begin
                errors = errors + 1;
                $display("ERROR backward config_error tile=%0d got=%0b expected=%0b",
                         checked_tiles, config_error, malformed_tlast);
            end
            checked_tiles = checked_tiles + 1;
            @(posedge clk);
            @(negedge clk);
        end
    endtask

    initial begin
        integer case_index;
        integer rows;
        integer inputs;
        integer outputs;
        integer dx_shift;
        integer dw_shift;

        seed = 32'h17bacc8d;
        seed = $urandom(seed);
        repeat (5) @(negedge clk);
        rst_n = 1'b1;

        run_case(8, 8, 8, 1, 0, 0, 1'b0);
        run_case(8, 8, 8, 2, 0, 0, 1'b0);
        run_case(8, 8, 8, 3, 3, 2, 1'b0);
        run_case(1, 1, 1, 0, 5, 7, 1'b0);
        for (case_index = 0; case_index < RANDOM_CASES; case_index = case_index + 1) begin
            rows = $urandom_range(8, 1);
            inputs = $urandom_range(8, 1);
            outputs = $urandom_range(8, 1);
            dx_shift = $urandom_range(12, 0);
            dw_shift = $urandom_range(12, 0);
            run_case(rows, inputs, outputs, 0, dx_shift, dw_shift, 1'b0);
        end

        // A premature TLAST must be reported while arithmetic remains exact.
        run_case(3, 5, 7, 0, 4, 3, 1'b1);

        run_exact_accumulation_modes();

        // Invalid dimensions and nonzero reserved bits complete without data.
        start_tile(0, 1, 1, 0, 0, 1'b0);
        while (!done) @(negedge clk);
        if (!config_error) begin
            errors = errors + 1;
            $display("ERROR backward accepted zero rows");
        end
        @(posedge clk);
        @(negedge clk);
        start_tile(1, 1, 1, 0, 0, 1'b1);
        while (!done) @(negedge clk);
        if (!config_error) begin
            errors = errors + 1;
            $display("ERROR backward accepted reserved command bits");
        end
        @(posedge clk);
        @(negedge clk);

        if (completed_tiles != checked_tiles || active_cycles == 0) begin
            errors = errors + 1;
            $display("ERROR backward counters tiles=%0d/%0d cycles=%0d",
                     completed_tiles, checked_tiles, active_cycles);
        end
        clear_counters = 1'b1;
        @(posedge clk);
        @(negedge clk);
        clear_counters = 1'b0;
        if (completed_tiles != 0 || active_cycles != 0) begin
            errors = errors + 1;
            $display("ERROR backward counters did not clear");
        end

        if (errors != 0) begin
            $display("tb_neuraedge_linear_backward8 FAILED errors=%0d tiles=%0d",
                     errors, checked_tiles);
            $finish;
        end
        $display("tb_neuraedge_linear_backward8 PASSED tiles=%0d random=%0d",
                 checked_tiles, RANDOM_CASES);
        $finish;
    end

    initial begin
        #10_000_000;
        $fatal(1, "tb_neuraedge_linear_backward8 watchdog timeout state=%0d", dut.state);
    end
endmodule

`default_nettype wire
