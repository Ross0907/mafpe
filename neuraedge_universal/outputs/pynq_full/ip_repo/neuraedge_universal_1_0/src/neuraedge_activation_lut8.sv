`timescale 1ns / 1ps
`default_nettype none

module neuraedge_activation_lut8 (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_table,
    input  wire load_valid,
    output logic load_ready,
    input  wire [7:0] load_address,
    input  wire signed [15:0] load_value,
    input  wire input_valid,
    output logic input_ready,
    input  wire [63:0] input_indices,
    output logic output_valid,
    input  wire output_ready,
    output logic [127:0] output_values,
    output logic config_error,
    output logic [8:0] loaded_entries
);
    // Replication gives eight independent lookup ports. A 256-entry table is
    // small enough for distributed RAM and preserves one eight-value vector per
    // cycle for GELU, SiLU, sigmoid, tanh, or a model-specific quantized curve.
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem0 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem1 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem2 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem3 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem4 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem5 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem6 [0:255];
    (* ram_style = "distributed" *) logic signed [15:0] lut_mem7 [0:255];
    logic [255:0] loaded_bitmap;
    logic load_count_pending_r;
    logic load_count_new_r;
    logic [1:0] result_fifo_count;
    logic result_fifo_write_pointer;
    logic result_fifo_read_pointer;
    logic [127:0] result_fifo_values [0:1];
    logic [7:0] result_fifo_loaded_masks [0:1];
    logic [63:0] lookup_indices_r;
    logic lookup_index_valid_r;
    logic [127:0] lookup_mid_values_r;
    logic [127:0] lookup_loaded_candidates_r;
    logic [31:0] lookup_high_nibbles_r;
    logic lookup_mid_valid_r;
    logic [127:0] lookup_result_r;
    logic [7:0] lookup_result_loaded_r;
    logic lookup_result_valid_r;
    logic [127:0] lookup_values_comb;
    logic [127:0] lookup_loaded_candidates_comb;
    wire signed [15:0] lookup_value0 = lut_mem0[lookup_indices_r[7:0]];
    wire signed [15:0] lookup_value1 = lut_mem1[lookup_indices_r[15:8]];
    wire signed [15:0] lookup_value2 = lut_mem2[lookup_indices_r[23:16]];
    wire signed [15:0] lookup_value3 = lut_mem3[lookup_indices_r[31:24]];
    wire signed [15:0] lookup_value4 = lut_mem4[lookup_indices_r[39:32]];
    wire signed [15:0] lookup_value5 = lut_mem5[lookup_indices_r[47:40]];
    wire signed [15:0] lookup_value6 = lut_mem6[lookup_indices_r[55:48]];
    wire signed [15:0] lookup_value7 = lut_mem7[lookup_indices_r[63:56]];
    wire result_fifo_pop = output_valid && output_ready;
    // Occupancy, rather than downstream ready, controls the lookup stages.
    // This deliberately permits one recovery bubble after a completely full
    // queue and prevents AXI backpressure from becoming a chip-wide data path.
    wire result_fifo_space = result_fifo_count != 2;
    wire result_fifo_push = lookup_result_valid_r && result_fifo_space;
    wire lookup_result_ready = !lookup_result_valid_r || result_fifo_space;
    wire lookup_mid_transfer = lookup_mid_valid_r && lookup_result_ready;
    wire lookup_mid_ready = !lookup_mid_valid_r || lookup_result_ready;
    wire lookup_index_transfer = lookup_index_valid_r && lookup_mid_ready;
    wire lookup_index_ready = !lookup_index_valid_r || lookup_mid_ready;
    wire lookup_input_accept = input_valid && input_ready;

    always_comb begin
        // The load and evaluate datapaths are independent at the acceptance
        // edge: a just-written entry is visible before a newly captured index
        // reaches the lookup stage. Keeping ready independent of peer valid
        // also prevents a valid->ready->controller combinational loop.
        load_ready = result_fifo_count == 0 && !lookup_index_valid_r
            && !lookup_mid_valid_r && !lookup_result_valid_r;
        input_ready = lookup_index_ready;
        output_valid = result_fifo_count != 0;
        output_values = result_fifo_values[result_fifo_read_pointer];
        config_error = output_valid
            && !(&result_fifo_loaded_masks[result_fifo_read_pointer]);

        lookup_values_comb = {lookup_value7, lookup_value6, lookup_value5,
                              lookup_value4, lookup_value3, lookup_value2,
                              lookup_value1, lookup_value0};
        for (int lane = 0; lane < 8; lane++) begin
            for (int group = 0; group < 16; group++) begin
                lookup_loaded_candidates_comb[lane*16+group] =
                    loaded_bitmap[group*16
                        + int'(lookup_indices_r[lane*8 +: 4])];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            result_fifo_count <= 2'd0;
            result_fifo_write_pointer <= 1'b0;
            result_fifo_read_pointer <= 1'b0;
            result_fifo_values[0] <= 128'd0;
            result_fifo_values[1] <= 128'd0;
            result_fifo_loaded_masks[0] <= 8'd0;
            result_fifo_loaded_masks[1] <= 8'd0;
            lookup_indices_r <= 64'd0;
            lookup_index_valid_r <= 1'b0;
            lookup_mid_values_r <= 128'd0;
            lookup_loaded_candidates_r <= 128'd0;
            lookup_high_nibbles_r <= 32'd0;
            lookup_mid_valid_r <= 1'b0;
            lookup_result_r <= 128'd0;
            lookup_result_loaded_r <= 8'd0;
            lookup_result_valid_r <= 1'b0;
            loaded_bitmap <= 256'd0;
            loaded_entries <= 9'd0;
            load_count_pending_r <= 1'b0;
            load_count_new_r <= 1'b0;
        end else if (clear_table) begin
            result_fifo_count <= 2'd0;
            result_fifo_write_pointer <= 1'b0;
            result_fifo_read_pointer <= 1'b0;
            lookup_index_valid_r <= 1'b0;
            lookup_mid_valid_r <= 1'b0;
            lookup_result_valid_r <= 1'b0;
            loaded_bitmap <= 256'd0;
            loaded_entries <= 9'd0;
            load_count_pending_r <= 1'b0;
            load_count_new_r <= 1'b0;
        end else begin
            load_count_pending_r <= 1'b0;
            if (load_count_pending_r && load_count_new_r) begin
                loaded_entries <= loaded_entries + 1'b1;
            end
            if (load_valid && load_ready) begin
                lut_mem0[load_address] <= load_value;
                lut_mem1[load_address] <= load_value;
                lut_mem2[load_address] <= load_value;
                lut_mem3[load_address] <= load_value;
                lut_mem4[load_address] <= load_value;
                lut_mem5[load_address] <= load_value;
                lut_mem6[load_address] <= load_value;
                lut_mem7[load_address] <= load_value;
                load_count_pending_r <= 1'b1;
                load_count_new_r <= !loaded_bitmap[load_address];
                loaded_bitmap[load_address] <= 1'b1;
            end

            if (lookup_input_accept) lookup_indices_r <= input_indices;
            unique case ({lookup_input_accept, lookup_index_transfer})
                2'b10: lookup_index_valid_r <= 1'b1;
                2'b01: lookup_index_valid_r <= 1'b0;
                default: begin end
            endcase

            if (lookup_index_transfer) begin
                lookup_mid_values_r <= lookup_values_comb;
                lookup_loaded_candidates_r <= lookup_loaded_candidates_comb;
                for (int lane = 0; lane < 8; lane++) begin
                    lookup_high_nibbles_r[lane*4 +: 4]
                        <= lookup_indices_r[lane*8+4 +: 4];
                end
            end
            unique case ({lookup_index_transfer, lookup_mid_transfer})
                2'b10: lookup_mid_valid_r <= 1'b1;
                2'b01: lookup_mid_valid_r <= 1'b0;
                default: begin end
            endcase

            if (lookup_mid_transfer) begin
                lookup_result_r <= lookup_mid_values_r;
                for (int lane = 0; lane < 8; lane++) begin
                    lookup_result_loaded_r[lane]
                        <= lookup_loaded_candidates_r[lane*16
                            + int'(lookup_high_nibbles_r[lane*4 +: 4])];
                end
            end
            unique case ({lookup_mid_transfer, result_fifo_push})
                2'b10: lookup_result_valid_r <= 1'b1;
                2'b01: lookup_result_valid_r <= 1'b0;
                default: begin end
            endcase

            if (result_fifo_push) begin
                for (int lane = 0; lane < 8; lane++) begin
                    result_fifo_values[result_fifo_write_pointer][lane*16 +: 16]
                        <= lookup_result_loaded_r[lane]
                        ? lookup_result_r[lane*16 +: 16] : 16'd0;
                end
                result_fifo_loaded_masks[result_fifo_write_pointer]
                    <= lookup_result_loaded_r;
                result_fifo_write_pointer <= !result_fifo_write_pointer;
            end

            if (result_fifo_pop) result_fifo_read_pointer <= !result_fifo_read_pointer;

            unique case ({result_fifo_push, result_fifo_pop})
                2'b10: result_fifo_count <= result_fifo_count + 1'b1;
                2'b01: result_fifo_count <= result_fifo_count - 1'b1;
                default: begin end
            endcase
        end
    end
endmodule

`default_nettype wire
