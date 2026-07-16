`timescale 1ns / 1ps
`default_nettype none

module neuraedge_lif_bank #(
    parameter int LANES = 32,
    parameter int LANES_PER_CYCLE = 8,
    parameter int MEMBRANE_BITS = 24,
    parameter int CURRENT_BITS = 16
) (
    input  wire clk,
    input  wire rst_n,
    input  wire clear_state,
    input  wire clear_counters,
    input  wire input_valid,
    output logic input_ready,
    input  wire [LANES*CURRENT_BITS-1:0] current_vector,
    input  wire [4:0] leak_shift,
    input  wire signed [MEMBRANE_BITS-1:0] threshold,
    input  wire signed [MEMBRANE_BITS-1:0] reset_value,
    output logic output_valid,
    input  wire output_ready,
    output logic [LANES-1:0] spikes,
    output logic [LANES*MEMBRANE_BITS-1:0] membrane_vector,
    output logic [LANES*MEMBRANE_BITS-1:0] pre_reset_membrane_vector,
    output logic [31:0] timestep_count,
    output logic [63:0] total_spike_count
);
    localparam int GROUPS = LANES / LANES_PER_CYCLE;

    typedef enum logic [3:0] {
        ST_IDLE = 4'd0,
        ST_GROUP_LOAD = 4'd1,
        ST_LEAK = 4'd2,
        ST_INTEGRATE = 4'd3,
        ST_SATURATE = 4'd4,
        ST_COMMIT = 4'd5,
        ST_OUTPUT = 4'd6,
        ST_LEAK_SUBTRACT = 4'd7,
        ST_THRESHOLD_COMPARE = 4'd8,
        ST_COUNT_LOW = 4'd9,
        ST_COUNT_WORD1 = 4'd10,
        ST_COUNT_WORD2 = 4'd11,
        ST_COUNT_WORD3 = 4'd12,
        ST_COUNT_DONE = 4'd13
    } state_t;

    state_t state;
    logic signed [MEMBRANE_BITS-1:0] membrane [0:LANES-1];
    logic signed [MEMBRANE_BITS-1:0] pre_reset_membrane [0:LANES-1];
    logic [LANES*CURRENT_BITS-1:0] current_vector_r;
    logic [4:0] leak_shift_r;
    logic signed [MEMBRANE_BITS-1:0] threshold_r;
    logic signed [MEMBRANE_BITS-1:0] reset_value_r;
    logic [$clog2(GROUPS)-1:0] group_index;
    logic [LANES-1:0] spike_work_r;
    logic [6:0] step_spike_count_r;
    logic [6:0] completed_step_spike_count_r;
    logic [16:0] counter_sum_r;
    logic signed [MEMBRANE_BITS-1:0] group_membrane_r [0:LANES_PER_CYCLE-1];
    logic signed [CURRENT_BITS-1:0] group_current_r [0:LANES_PER_CYCLE-1];
    logic signed [MEMBRANE_BITS:0] leak_shifted_r [0:LANES_PER_CYCLE-1];
    logic signed [MEMBRANE_BITS:0] leaked_r [0:LANES_PER_CYCLE-1];
    logic signed [MEMBRANE_BITS:0] integrated_r [0:LANES_PER_CYCLE-1];
    logic signed [MEMBRANE_BITS-1:0] saturated_r [0:LANES_PER_CYCLE-1];
    logic [LANES_PER_CYCLE-1:0] group_spikes_r;

    function automatic logic signed [MEMBRANE_BITS-1:0] saturate_membrane(
        input logic signed [MEMBRANE_BITS:0] value
    );
        begin
            unique case (value[MEMBRANE_BITS:MEMBRANE_BITS-1])
                2'b01: saturate_membrane = {1'b0, {(MEMBRANE_BITS-1){1'b1}}};
                2'b10: saturate_membrane = {1'b1, {(MEMBRANE_BITS-1){1'b0}}};
                default: saturate_membrane = value[MEMBRANE_BITS-1:0];
            endcase
        end
    endfunction

    function automatic logic [3:0] popcount8(input logic [7:0] value);
        logic [1:0] pair0;
        logic [1:0] pair1;
        logic [1:0] pair2;
        logic [1:0] pair3;
        logic [2:0] half0;
        logic [2:0] half1;
        begin
            pair0 = value[0] + value[1];
            pair1 = value[2] + value[3];
            pair2 = value[4] + value[5];
            pair3 = value[6] + value[7];
            half0 = pair0 + pair1;
            half1 = pair2 + pair3;
            popcount8 = half0 + half1;
        end
    endfunction

    always_comb begin
        input_ready = state == ST_IDLE;
        output_valid = state == ST_OUTPUT;
        for (int lane = 0; lane < LANES; lane++) begin
            membrane_vector[lane*MEMBRANE_BITS +: MEMBRANE_BITS] = membrane[lane];
            pre_reset_membrane_vector[lane*MEMBRANE_BITS +: MEMBRANE_BITS]
                = pre_reset_membrane[lane];
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            current_vector_r <= '0;
            leak_shift_r <= 5'd0;
            threshold_r <= '0;
            reset_value_r <= '0;
            group_index <= '0;
            spike_work_r <= '0;
            step_spike_count_r <= 7'd0;
            completed_step_spike_count_r <= 7'd0;
            counter_sum_r <= 17'd0;
            group_spikes_r <= '0;
            spikes <= '0;
            timestep_count <= 32'd0;
            total_spike_count <= 64'd0;
            for (int lane = 0; lane < LANES; lane++) begin
                membrane[lane] <= '0;
                pre_reset_membrane[lane] <= '0;
            end
            for (int lane = 0; lane < LANES_PER_CYCLE; lane++) begin
                group_membrane_r[lane] <= '0;
                group_current_r[lane] <= '0;
                leak_shifted_r[lane] <= '0;
                leaked_r[lane] <= '0;
                integrated_r[lane] <= '0;
                saturated_r[lane] <= '0;
            end
        end else if (clear_state) begin
            state <= ST_IDLE;
            spike_work_r <= '0;
            spikes <= '0;
            timestep_count <= 32'd0;
            for (int lane = 0; lane < LANES; lane++) begin
                membrane[lane] <= reset_value;
                pre_reset_membrane[lane] <= reset_value;
            end
        end else begin
            unique case (state)
                ST_IDLE: begin
                    if (input_valid) begin
                        current_vector_r <= current_vector;
                        leak_shift_r <= leak_shift;
                        threshold_r <= threshold;
                        reset_value_r <= reset_value;
                        group_index <= '0;
                        spike_work_r <= '0;
                        step_spike_count_r <= 7'd0;
                        state <= ST_GROUP_LOAD;
                    end
                end

                ST_GROUP_LOAD: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        logic [$clog2(LANES)-1:0] lane;
                        lane = $clog2(LANES)'(
                            group_index * LANES_PER_CYCLE + local_lane);
                        group_membrane_r[local_lane] <= membrane[lane];
                        group_current_r[local_lane] <= $signed(
                            current_vector_r[lane*CURRENT_BITS +: CURRENT_BITS]);
                    end
                    state <= ST_LEAK;
                end

                ST_LEAK: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        logic signed [MEMBRANE_BITS:0] membrane_extended;
                        membrane_extended = {
                            group_membrane_r[local_lane][MEMBRANE_BITS-1],
                            group_membrane_r[local_lane]};
                        leak_shifted_r[local_lane] <=
                            membrane_extended >>> leak_shift_r;
                    end
                    state <= ST_LEAK_SUBTRACT;
                end

                ST_LEAK_SUBTRACT: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        logic signed [MEMBRANE_BITS:0] membrane_extended;
                        membrane_extended = {
                            group_membrane_r[local_lane][MEMBRANE_BITS-1],
                            group_membrane_r[local_lane]};
                        leaked_r[local_lane] <= membrane_extended
                            - leak_shifted_r[local_lane];
                    end
                    state <= ST_INTEGRATE;
                end

                ST_INTEGRATE: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        integrated_r[local_lane] <= leaked_r[local_lane]
                            + $signed({{(MEMBRANE_BITS+1-CURRENT_BITS){
                                group_current_r[local_lane][CURRENT_BITS-1]}},
                                group_current_r[local_lane]});
                    end
                    state <= ST_SATURATE;
                end

                ST_SATURATE: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        saturated_r[local_lane] <=
                            saturate_membrane(integrated_r[local_lane]);
                    end
                    state <= ST_THRESHOLD_COMPARE;
                end

                ST_THRESHOLD_COMPARE: begin
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        group_spikes_r[local_lane] <=
                            saturated_r[local_lane] >= threshold_r;
                    end
                    state <= ST_COMMIT;
                end

                ST_COMMIT: begin
                    logic [LANES-1:0] next_spikes;
                    logic [3:0] group_spike_count;
                    next_spikes = spike_work_r;
                    for (int local_lane = 0; local_lane < LANES_PER_CYCLE; local_lane++) begin
                        logic [$clog2(LANES)-1:0] lane;
                        lane = $clog2(LANES)'(
                            group_index * LANES_PER_CYCLE + local_lane);
                        pre_reset_membrane[lane] <= saturated_r[local_lane];
                        if (group_spikes_r[local_lane]) begin
                            membrane[lane] <= reset_value_r;
                        end else begin
                            membrane[lane] <= saturated_r[local_lane];
                        end
                    end
                    next_spikes[group_index*LANES_PER_CYCLE +: LANES_PER_CYCLE]
                        = group_spikes_r;
                    group_spike_count = popcount8(group_spikes_r);
                    spike_work_r <= next_spikes;
                    step_spike_count_r <= step_spike_count_r
                        + {3'd0, group_spike_count};

                    if (group_index == $clog2(GROUPS)'(GROUPS - 1)) begin
                        spikes <= next_spikes;
                        timestep_count <= timestep_count + 1'b1;
                        completed_step_spike_count_r <= step_spike_count_r
                            + {3'd0, group_spike_count};
                        state <= ST_COUNT_LOW;
                    end else begin
                        group_index <= group_index + 1'b1;
                        state <= ST_GROUP_LOAD;
                    end
                end

                // The cumulative diagnostic counter is deliberately updated
                // as four 16-bit words. A monolithic 64-bit addition placed a
                // popcount and long carry chain on the 200 MHz neuron path.
                ST_COUNT_LOW: begin
                    counter_sum_r <= {1'b0, total_spike_count[15:0]}
                        + {10'd0, completed_step_spike_count_r};
                    state <= ST_COUNT_WORD1;
                end

                ST_COUNT_WORD1: begin
                    total_spike_count[15:0] <= counter_sum_r[15:0];
                    counter_sum_r <= {1'b0, total_spike_count[31:16]}
                        + {16'd0, counter_sum_r[16]};
                    state <= ST_COUNT_WORD2;
                end

                ST_COUNT_WORD2: begin
                    total_spike_count[31:16] <= counter_sum_r[15:0];
                    counter_sum_r <= {1'b0, total_spike_count[47:32]}
                        + {16'd0, counter_sum_r[16]};
                    state <= ST_COUNT_WORD3;
                end

                ST_COUNT_WORD3: begin
                    total_spike_count[47:32] <= counter_sum_r[15:0];
                    counter_sum_r <= {1'b0, total_spike_count[63:48]}
                        + {16'd0, counter_sum_r[16]};
                    state <= ST_COUNT_DONE;
                end

                ST_COUNT_DONE: begin
                    total_spike_count[63:48] <= counter_sum_r[15:0];
                    state <= ST_OUTPUT;
                end

                ST_OUTPUT: if (output_ready) state <= ST_IDLE;
                default: state <= ST_IDLE;
            endcase
            if (clear_counters) total_spike_count <= 64'd0;
        end
    end

    initial begin
        if (LANES != 32) $fatal(1, "neuraedge_lif_bank currently requires LANES=32");
        if (LANES_PER_CYCLE != 8) $fatal(1, "neuraedge_lif_bank currently requires LANES_PER_CYCLE=8");
        if (LANES % LANES_PER_CYCLE != 0) $fatal(1, "LANES_PER_CYCLE must divide LANES");
        if (CURRENT_BITS >= MEMBRANE_BITS) $fatal(1, "CURRENT_BITS must be smaller than MEMBRANE_BITS");
    end
endmodule

`default_nettype wire
