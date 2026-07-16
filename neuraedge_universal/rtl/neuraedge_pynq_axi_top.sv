`timescale 1ns / 1ps
`default_nettype none

module neuraedge_pynq_axi_top #(
    parameter int AXI_ADDR_BITS = 8
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aclk, ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn, FREQ_HZ 200000000" *)
    input  wire aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME aresetn, POLARITY ACTIVE_LOW" *)
    input  wire aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 8, FREQ_HZ 200000000, HAS_BURST 0, HAS_LOCK 0, HAS_PROT 0, HAS_CACHE 0, HAS_QOS 0, HAS_REGION 0, SUPPORTS_NARROW_BURST 0" *)
    input  wire [AXI_ADDR_BITS-1:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output logic s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output logic s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output logic [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output logic s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [AXI_ADDR_BITS-1:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output logic s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output logic [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output logic [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output logic s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS, TDATA_NUM_BYTES 8, HAS_TKEEP 1, HAS_TLAST 1, FREQ_HZ 200000000" *)
    input  wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output logic s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input  wire [63:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input  wire [7:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input  wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS, TDATA_NUM_BYTES 8, HAS_TKEEP 1, HAS_TLAST 1, FREQ_HZ 200000000" *)
    output logic m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output logic [63:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output logic [7:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output logic m_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 interrupt INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME interrupt, SENSITIVITY LEVEL_HIGH" *)
    output logic interrupt
);
    localparam logic [7:0] ADDR_CONTROL = 8'h00;
    localparam logic [7:0] ADDR_STATUS = 8'h04;
    localparam logic [7:0] ADDR_ACTIVE_OPCODE = 8'h08;
    localparam logic [7:0] ADDR_COMMAND_COUNT = 8'h0c;
    localparam logic [7:0] ADDR_COMMAND0 = 8'h10;
    localparam logic [7:0] ADDR_COMMAND1 = 8'h14;
    localparam logic [7:0] ADDR_COMMAND2 = 8'h18;
    localparam logic [7:0] ADDR_COMMAND3 = 8'h1c;
    localparam logic [7:0] ADDR_INPUT_BEATS = 8'h20;
    localparam logic [7:0] ADDR_OUTPUT_BEATS = 8'h24;
    localparam logic [7:0] ADDR_TARGET_CLOCK = 8'h28;
    localparam logic [7:0] ADDR_CAPABILITIES = 8'h2c;
    localparam logic [7:0] ADDR_VERSION = 8'h30;
    localparam logic [7:0] ADDR_ID = 8'h34;
    localparam logic [7:0] ADDR_TENSOR_LOADED = 8'h38;
    localparam logic [7:0] ADDR_TENSOR_ACTIVE = 8'h3c;
    localparam logic [7:0] ADDR_TENSOR_CYCLES = 8'h40;
    localparam logic [7:0] ADDR_TENSOR_SKIPPED = 8'h44;
    localparam logic [7:0] ADDR_DENSE_MACS_LOW = 8'h48;
    localparam logic [7:0] ADDR_DENSE_MACS_HIGH = 8'h4c;
    localparam logic [7:0] ADDR_SPIKE_ADDS_LOW = 8'h50;
    localparam logic [7:0] ADDR_SPIKE_ADDS_HIGH = 8'h54;
    localparam logic [7:0] ADDR_LAYERNORM_CYCLES = 8'h58;
    localparam logic [7:0] ADDR_LAYERNORM_MEAN = 8'h5c;
    localparam logic [7:0] ADDR_LAYERNORM_VARIANCE = 8'h60;
    localparam logic [7:0] ADDR_LAYERNORM_SIGMA = 8'h64;
    localparam logic [7:0] ADDR_SOFTMAX_CYCLES = 8'h68;
    localparam logic [7:0] ADDR_SOFTMAX_MAXIMUM = 8'h6c;
    localparam logic [7:0] ADDR_SOFTMAX_DENOMINATOR = 8'h70;
    localparam logic [7:0] ADDR_LIF_SPIKES_LOW = 8'h74;
    localparam logic [7:0] ADDR_LIF_SPIKES_HIGH = 8'h78;
    localparam logic [7:0] ADDR_OPTIMIZER_UPDATES = 8'h7c;
    localparam logic [7:0] ADDR_LUT_ENTRIES = 8'h80;
    localparam logic [7:0] ADDR_LAYERNORM_RECIPROCAL = 8'h84;
    localparam logic [7:0] ADDR_SOFTMAX_RECIPROCAL = 8'h88;
    localparam logic [7:0] ADDR_PROGRAM_RECORDS = 8'h8c;
    localparam logic [7:0] ADDR_PROGRAM_OUTPUT_BEATS = 8'h90;
    localparam logic [7:0] ADDR_TRAINING_TILES = 8'h94;
    localparam logic [7:0] ADDR_TRAINING_CYCLES = 8'h98;

    logic [31:0] command_register [0:3];
    logic [127:0] submitted_command;
    logic command_pending;
    logic interrupt_enable;
    logic done_sticky;
    logic wrapper_error;
    logic program_start_r;
    logic program_clear_r;
    logic [31:0] program_output_beats_r;
    wire program_active;
    wire program_done;
    wire program_error;
    wire [31:0] program_records;

    logic aw_hold;
    logic [AXI_ADDR_BITS-1:0] awaddr_hold;
    logic w_hold;
    logic [31:0] wdata_hold;
    logic [3:0] wstrb_hold;

    wire core_command_ready;
    wire core_command_done;
    wire sequencer_core_command_valid;
    wire [127:0] sequencer_core_command_data;
    wire sequencer_core_s_valid;
    wire [63:0] sequencer_core_s_data;
    wire sequencer_core_s_last;
    wire sequencer_core_m_ready;
    wire core_busy;
    wire core_error;
    wire [3:0] core_active_opcode;
    wire [31:0] core_command_count;
    wire [31:0] core_input_beats;
    wire [31:0] core_output_beats;
    wire [11:0] core_tensor_loaded_activations;
    wire [11:0] core_tensor_loaded_weights;
    wire [11:0] core_tensor_active_issues;
    wire [31:0] core_tensor_compute_cycles;
    wire [31:0] core_tensor_skipped_issues;
    wire [63:0] core_tensor_dense_macs;
    wire [63:0] core_tensor_spike_adds;
    wire [31:0] core_layernorm_cycles;
    wire signed [31:0] core_layernorm_mean;
    wire [31:0] core_layernorm_variance;
    wire [31:0] core_layernorm_sigma;
    wire [31:0] core_layernorm_reciprocal;
    wire [31:0] core_softmax_cycles;
    wire signed [23:0] core_softmax_maximum;
    wire [23:0] core_softmax_denominator;
    wire [31:0] core_softmax_reciprocal;
    wire [63:0] core_lif_total_spikes;
    wire [31:0] core_optimizer_updates;
    wire [8:0] core_lut_loaded_entries;
    wire [31:0] core_training_tiles;
    wire [31:0] core_training_cycles;
    logic core_s_valid;
    wire core_s_ready;
    wire core_raw_s_ready;
    logic core_s_ready_r;
    logic next_core_s_valid;
    logic next_core_s_ready;
    logic [63:0] core_s_data;
    logic core_s_last;
    logic [1:0] input_fifo_count;
    logic input_fifo_write_pointer;
    logic input_fifo_read_pointer;
    logic [63:0] input_fifo_data [0:1];
    logic input_fifo_last [0:1];
    logic core_input_capture_valid;
    logic [63:0] core_input_capture_data;
    logic core_input_capture_last;
    wire core_raw_m_valid;
    wire core_raw_m_ready;
    logic core_raw_m_ready_r;
    wire [63:0] core_raw_m_data;
    wire core_raw_m_last;
    logic core_m_valid;
    logic next_core_m_valid;
    logic next_core_raw_m_ready;
    wire core_m_ready;
    wire sequencer_s_ready;
    wire sequencer_m_valid;
    wire [63:0] sequencer_m_data;
    wire sequencer_m_last;
    logic core_m_ready_r;
    logic [63:0] core_m_data;
    logic core_m_last;
    logic core_output_capture_valid;
    logic [63:0] core_output_capture_data;
    logic core_output_capture_last;
    logic [1:0] output_fifo_count;
    logic output_fifo_write_pointer;
    logic output_fifo_read_pointer;
    logic [63:0] output_fifo_data [0:1];
    logic output_fifo_last [0:1];
    logic output_register_valid_r;
    logic [63:0] output_register_data_r;
    logic output_register_last_r;
    logic [1:0] next_output_fifo_count;
    logic next_core_output_capture_valid;
    logic next_core_m_ready;

    wire input_fifo_push = s_axis_tvalid && s_axis_tready;
    wire input_fifo_pop = input_fifo_count != 0 && !core_input_capture_valid;
    wire wrapper_input_ready = program_active ? sequencer_s_ready : core_s_ready;
    wire wrapper_output_valid = program_active ? sequencer_m_valid : core_m_valid;
    wire [63:0] wrapper_output_data = program_active
        ? sequencer_m_data : core_m_data;
    wire wrapper_output_last = program_active ? sequencer_m_last : core_m_last;
    wire core_output_capture_push = wrapper_output_valid && core_m_ready_r;
    wire core_output_capture_pop = core_output_capture_valid
        && output_fifo_count != 2;
    wire output_fifo_push = core_output_capture_pop;
    wire output_register_ready = !output_register_valid_r || m_axis_tready;
    wire output_fifo_pop = output_fifo_count != 0 && output_register_ready;
    wire selected_core_s_valid = program_active
        ? sequencer_core_s_valid : core_input_capture_valid;
    wire [63:0] selected_core_s_data = program_active
        ? sequencer_core_s_data : core_input_capture_data;
    wire selected_core_s_last = program_active
        ? sequencer_core_s_last : core_input_capture_last;
    wire core_s_boundary_push = selected_core_s_valid && core_s_ready_r;
    wire core_s_boundary_pop = core_s_valid && core_raw_s_ready;
    wire core_m_boundary_push = core_raw_m_valid && core_raw_m_ready_r;
    wire core_m_boundary_pop = core_m_valid && core_m_ready;
    assign core_s_ready = core_s_ready_r;
    assign core_m_ready = program_active
        ? sequencer_core_m_ready : core_m_ready_r;
    assign core_raw_m_ready = core_raw_m_ready_r;

    always_comb begin
        next_core_s_valid = core_s_valid;
        unique case ({core_s_boundary_push, core_s_boundary_pop})
            2'b10: next_core_s_valid = 1'b1;
            2'b01: next_core_s_valid = 1'b0;
            default: begin end
        endcase
        next_core_s_ready = !next_core_s_valid || core_raw_s_ready;
    end

    always_comb begin
        next_core_m_valid = core_m_valid;
        unique case ({core_m_boundary_push, core_m_boundary_pop})
            2'b10: next_core_m_valid = 1'b1;
            2'b01: next_core_m_valid = 1'b0;
            default: begin end
        endcase
        next_core_raw_m_ready = !next_core_m_valid || core_m_ready;
    end

    // Registered credit prevents a child engine's READY state from becoming
    // a payload clock-enable while retaining simultaneous consume/refill.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            core_s_valid <= 1'b0;
            core_s_ready_r <= 1'b1;
            core_s_data <= 64'd0;
            core_s_last <= 1'b0;
        end else begin
            core_s_valid <= next_core_s_valid;
            core_s_ready_r <= next_core_s_ready;
            if (core_s_ready_r) begin
                core_s_data <= selected_core_s_data;
                core_s_last <= selected_core_s_last;
            end
        end
    end

    // This elastic boundary isolates every child engine's result-valid path
    // from the descriptor and wrapper FIFOs while retaining one beat/cycle.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            core_m_valid <= 1'b0;
            core_raw_m_ready_r <= 1'b1;
            core_m_data <= 64'd0;
            core_m_last <= 1'b0;
        end else begin
            core_m_valid <= next_core_m_valid;
            core_raw_m_ready_r <= next_core_raw_m_ready;
            if (core_raw_m_ready_r) begin
                core_m_data <= core_raw_m_data;
                core_m_last <= core_raw_m_last;
            end
        end
    end

    always_comb begin
        next_output_fifo_count = output_fifo_count;
        unique case ({output_fifo_push, output_fifo_pop})
            2'b10: next_output_fifo_count = output_fifo_count + 1'b1;
            2'b01: next_output_fifo_count = output_fifo_count - 1'b1;
            default: begin end
        endcase

        next_core_output_capture_valid = core_output_capture_valid;
        unique case ({core_output_capture_push, core_output_capture_pop})
            2'b10: next_core_output_capture_valid = 1'b1;
            2'b01: next_core_output_capture_valid = 1'b0;
            default: begin end
        endcase
        // This next-cycle credit is conservative when a full FIFO pops: the
        // resulting single restart bubble prevents any combinational READY
        // path while preserving one result per cycle during steady traffic.
        next_core_m_ready = !next_core_output_capture_valid
            || next_output_fifo_count != 2;
    end

    // Two entries isolate external backpressure from every internal engine while
    // still allowing simultaneous enqueue/dequeue at one result per clock.
    function automatic logic [31:0] apply_write_strobes(
        input logic [31:0] old_value,
        input logic [31:0] new_value,
        input logic [3:0] strobes
    );
        logic [31:0] merged;
        begin
            merged = old_value;
            for (int byte_index = 0; byte_index < 4; byte_index++) begin
                if (strobes[byte_index]) begin
                    merged[byte_index*8 +: 8] = new_value[byte_index*8 +: 8];
                end
            end
            apply_write_strobes = merged;
        end
    endfunction

    function automatic logic [31:0] register_read(input logic [7:0] address);
        begin
            unique case (address)
                ADDR_CONTROL: register_read = {
                    23'd0, interrupt_enable, 5'd0, program_active,
                    1'b0, command_pending};
                ADDR_STATUS: register_read = {
                    26'd0, program_active, interrupt, command_pending,
                    done_sticky, core_error || wrapper_error || program_error,
                    core_busy || command_pending || program_active};
                ADDR_ACTIVE_OPCODE: register_read = {28'd0, core_active_opcode};
                ADDR_COMMAND_COUNT: register_read = core_command_count;
                ADDR_COMMAND0: register_read = command_register[0];
                ADDR_COMMAND1: register_read = command_register[1];
                ADDR_COMMAND2: register_read = command_register[2];
                ADDR_COMMAND3: register_read = command_register[3];
                ADDR_INPUT_BEATS: register_read = core_input_beats;
                ADDR_OUTPUT_BEATS: register_read = core_output_beats;
                ADDR_TARGET_CLOCK: register_read = 32'd200000000;
                // Bits correspond to tensor, binary/ternary, alpha-XNOR,
                // vector/LUT, normalization, LIF, optimizer, and backward tile.
                ADDR_CAPABILITIES: register_read = 32'h000001ff;
                ADDR_VERSION: register_read = 32'h00010007;
                ADDR_ID: register_read = 32'h4e455552;
                ADDR_TENSOR_LOADED: register_read = {
                    4'd0, core_tensor_loaded_weights,
                    4'd0, core_tensor_loaded_activations};
                ADDR_TENSOR_ACTIVE: register_read = {20'd0, core_tensor_active_issues};
                ADDR_TENSOR_CYCLES: register_read = core_tensor_compute_cycles;
                ADDR_TENSOR_SKIPPED: register_read = core_tensor_skipped_issues;
                ADDR_DENSE_MACS_LOW: register_read = core_tensor_dense_macs[31:0];
                ADDR_DENSE_MACS_HIGH: register_read = core_tensor_dense_macs[63:32];
                ADDR_SPIKE_ADDS_LOW: register_read = core_tensor_spike_adds[31:0];
                ADDR_SPIKE_ADDS_HIGH: register_read = core_tensor_spike_adds[63:32];
                ADDR_LAYERNORM_CYCLES: register_read = core_layernorm_cycles;
                ADDR_LAYERNORM_MEAN: register_read = core_layernorm_mean;
                ADDR_LAYERNORM_VARIANCE: register_read = core_layernorm_variance;
                ADDR_LAYERNORM_SIGMA: register_read = core_layernorm_sigma;
                ADDR_SOFTMAX_CYCLES: register_read = core_softmax_cycles;
                ADDR_SOFTMAX_MAXIMUM: register_read =
                    {{8{core_softmax_maximum[23]}}, core_softmax_maximum};
                ADDR_SOFTMAX_DENOMINATOR: register_read =
                    {8'd0, core_softmax_denominator};
                ADDR_LIF_SPIKES_LOW: register_read = core_lif_total_spikes[31:0];
                ADDR_LIF_SPIKES_HIGH: register_read = core_lif_total_spikes[63:32];
                ADDR_OPTIMIZER_UPDATES: register_read = core_optimizer_updates;
                ADDR_LUT_ENTRIES: register_read = {23'd0, core_lut_loaded_entries};
                ADDR_LAYERNORM_RECIPROCAL: register_read = core_layernorm_reciprocal;
                ADDR_SOFTMAX_RECIPROCAL: register_read = core_softmax_reciprocal;
                ADDR_PROGRAM_RECORDS: register_read = program_records;
                ADDR_PROGRAM_OUTPUT_BEATS: register_read = program_output_beats_r;
                ADDR_TRAINING_TILES: register_read = core_training_tiles;
                ADDR_TRAINING_CYCLES: register_read = core_training_cycles;
                default: register_read = 32'd0;
            endcase
        end
    endfunction

    neuraedge_descriptor_sequencer u_descriptor_sequencer (
        .clk(aclk), .rst_n(aresetn),
        .program_start(program_start_r),
        .clear_status(program_clear_r),
        .program_output_beats(program_output_beats_r),
        .program_active(program_active), .program_done(program_done),
        .program_error(program_error), .program_records(program_records),
        .s_axis_tvalid(core_input_capture_valid),
        .s_axis_tready(sequencer_s_ready),
        .s_axis_tdata(core_input_capture_data),
        .s_axis_tlast(core_input_capture_last),
        .m_axis_tvalid(sequencer_m_valid),
        .m_axis_tready(core_m_ready_r),
        .m_axis_tdata(sequencer_m_data),
        .m_axis_tlast(sequencer_m_last),
        .core_command_valid(sequencer_core_command_valid),
        .core_command_ready(core_command_ready),
        .core_command_data(sequencer_core_command_data),
        .core_command_done(core_command_done),
        .core_status_error(core_error),
        .core_s_valid(sequencer_core_s_valid), .core_s_ready(core_s_ready),
        .core_s_data(sequencer_core_s_data), .core_s_last(sequencer_core_s_last),
        .core_m_valid(core_m_valid), .core_m_ready(sequencer_core_m_ready),
        .core_m_data(core_m_data), .core_m_last(core_m_last)
    );

    neuraedge_universal_core u_core (
        .clk(aclk), .rst_n(aresetn),
        .command_valid(program_active
            ? sequencer_core_command_valid : command_pending),
        .command_ready(core_command_ready),
        .command_data(program_active
            ? sequencer_core_command_data : submitted_command),
        .command_done(core_command_done),
        .status_busy(core_busy), .status_error(core_error),
        .status_active_opcode(core_active_opcode),
        .status_command_count(core_command_count),
        .status_input_beats(core_input_beats), .status_output_beats(core_output_beats),
        .debug_tensor_loaded_activations(core_tensor_loaded_activations),
        .debug_tensor_loaded_weights(core_tensor_loaded_weights),
        .debug_tensor_active_issues(core_tensor_active_issues),
        .debug_tensor_compute_cycles(core_tensor_compute_cycles),
        .debug_tensor_skipped_issues(core_tensor_skipped_issues),
        .debug_tensor_dense_macs(core_tensor_dense_macs),
        .debug_tensor_spike_adds(core_tensor_spike_adds),
        .debug_layernorm_cycles(core_layernorm_cycles),
        .debug_layernorm_mean(core_layernorm_mean),
        .debug_layernorm_variance(core_layernorm_variance),
        .debug_layernorm_sigma(core_layernorm_sigma),
        .debug_layernorm_reciprocal(core_layernorm_reciprocal),
        .debug_softmax_cycles(core_softmax_cycles),
        .debug_softmax_maximum(core_softmax_maximum),
        .debug_softmax_denominator(core_softmax_denominator),
        .debug_softmax_reciprocal(core_softmax_reciprocal),
        .debug_lif_total_spikes(core_lif_total_spikes),
        .debug_optimizer_updates(core_optimizer_updates),
        .debug_lut_loaded_entries(core_lut_loaded_entries),
        .debug_training_tiles(core_training_tiles),
        .debug_training_cycles(core_training_cycles),
        .s_axis_tvalid(core_s_valid), .s_axis_tready(core_raw_s_ready),
        .s_axis_tdata(core_s_data), .s_axis_tlast(core_s_last),
        .m_axis_tvalid(core_raw_m_valid), .m_axis_tready(core_raw_m_ready),
        .m_axis_tdata(core_raw_m_data), .m_axis_tlast(core_raw_m_last)
    );

    always_comb begin
        // A two-entry elastic FIFO makes the external ready path local while
        // still accepting one input beat per clock during simultaneous pop/push.
        s_axis_tready = input_fifo_count != 2;
        s_axi_awready = !aw_hold && !s_axi_bvalid;
        s_axi_wready = !w_hold && !s_axi_bvalid;
        s_axi_bresp = 2'b00;
        s_axi_arready = !s_axi_rvalid;
        s_axi_rresp = 2'b00;
        m_axis_tvalid = output_register_valid_r;
        m_axis_tdata = output_register_data_r;
        m_axis_tkeep = 8'hff;
        m_axis_tlast = output_register_last_r;
        interrupt = interrupt_enable && done_sticky;
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            for (int word_index = 0; word_index < 4; word_index++) begin
                command_register[word_index] <= 32'd0;
            end
            submitted_command <= 128'd0;
            command_pending <= 1'b0;
            interrupt_enable <= 1'b0;
            done_sticky <= 1'b0;
            wrapper_error <= 1'b0;
            program_start_r <= 1'b0;
            program_clear_r <= 1'b0;
            program_output_beats_r <= 32'd0;
            aw_hold <= 1'b0;
            awaddr_hold <= '0;
            w_hold <= 1'b0;
            wdata_hold <= 32'd0;
            wstrb_hold <= 4'd0;
            s_axi_bvalid <= 1'b0;
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= 32'd0;
            output_fifo_count <= 2'd0;
            core_output_capture_valid <= 1'b0;
            core_m_ready_r <= 1'b1;
            core_output_capture_data <= 64'd0;
            core_output_capture_last <= 1'b0;
            output_fifo_write_pointer <= 1'b0;
            output_fifo_read_pointer <= 1'b0;
            output_fifo_data[0] <= 64'd0;
            output_fifo_data[1] <= 64'd0;
            output_fifo_last[0] <= 1'b0;
            output_fifo_last[1] <= 1'b0;
            output_register_valid_r <= 1'b0;
            output_register_data_r <= 64'd0;
            output_register_last_r <= 1'b0;
            input_fifo_count <= 2'd0;
            input_fifo_write_pointer <= 1'b0;
            input_fifo_read_pointer <= 1'b0;
            input_fifo_data[0] <= 64'd0;
            input_fifo_data[1] <= 64'd0;
            input_fifo_last[0] <= 1'b0;
            input_fifo_last[1] <= 1'b0;
            core_input_capture_valid <= 1'b0;
            core_input_capture_data <= 64'd0;
            core_input_capture_last <= 1'b0;
        end else begin
            program_start_r <= 1'b0;
            program_clear_r <= 1'b0;
            // The prefetch stage decouples payload storage from deep core-ready
            // logic. At 200 MHz it accepts one 64-bit beat every two cycles.
            if (input_fifo_pop) begin
                core_input_capture_data <= input_fifo_data[input_fifo_read_pointer];
                core_input_capture_last <= input_fifo_last[input_fifo_read_pointer];
                core_input_capture_valid <= 1'b1;
            end else if (core_input_capture_valid && wrapper_input_ready) begin
                core_input_capture_valid <= 1'b0;
            end
            if (input_fifo_push) begin
                input_fifo_data[input_fifo_write_pointer] <= s_axis_tdata;
                input_fifo_last[input_fifo_write_pointer] <= s_axis_tlast;
                input_fifo_write_pointer <= !input_fifo_write_pointer;
            end
            if (input_fifo_pop) begin
                input_fifo_read_pointer <= !input_fifo_read_pointer;
            end
            unique case ({input_fifo_push, input_fifo_pop})
                2'b10: input_fifo_count <= input_fifo_count + 1'b1;
                2'b01: input_fifo_count <= input_fifo_count - 1'b1;
                default: begin end
            endcase
            // Sampling an invalid payload is harmless because the separate
            // valid bit qualifies it. Enabling on local registered credit,
            // rather than deep VALID && READY logic, keeps all 65 payload
            // clock-enables local to this wrapper.
            if (core_m_ready_r) begin
                core_output_capture_data <= wrapper_output_data;
                core_output_capture_last <= wrapper_output_last;
            end
            core_output_capture_valid <= next_core_output_capture_valid;
            core_m_ready_r <= next_core_m_ready;
            if (output_fifo_push) begin
                output_fifo_data[output_fifo_write_pointer]
                    <= core_output_capture_data;
                output_fifo_last[output_fifo_write_pointer]
                    <= core_output_capture_last;
                output_fifo_write_pointer <= !output_fifo_write_pointer;
            end
            if (output_fifo_pop) begin
                output_fifo_read_pointer <= !output_fifo_read_pointer;
            end
            output_fifo_count <= next_output_fifo_count;
            if (output_register_ready) begin
                output_register_valid_r <= output_fifo_count != 0;
                if (output_fifo_count != 0) begin
                    output_register_data_r
                        <= output_fifo_data[output_fifo_read_pointer];
                    output_register_last_r
                        <= output_fifo_last[output_fifo_read_pointer];
                end
            end
            if (command_pending && !program_active && core_command_ready) begin
                command_pending <= 1'b0;
            end
            if ((!program_active && core_command_done) || program_done) begin
                done_sticky <= 1'b1;
            end
            if (input_fifo_push && s_axis_tkeep != 8'hff) begin
                wrapper_error <= 1'b1;
            end

            if (s_axi_awvalid && s_axi_awready) begin
                aw_hold <= 1'b1;
                awaddr_hold <= s_axi_awaddr;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_hold <= 1'b1;
                wdata_hold <= s_axi_wdata;
                wstrb_hold <= s_axi_wstrb;
            end

            if (aw_hold && w_hold && !s_axi_bvalid) begin
                unique case (awaddr_hold[7:0])
                    ADDR_CONTROL: begin
                        if (wstrb_hold[1]) interrupt_enable <= wdata_hold[8];
                        if (wstrb_hold[0] && wdata_hold[1]) begin
                            done_sticky <= 1'b0;
                            wrapper_error <= 1'b0;
                            program_clear_r <= 1'b1;
                        end
                        if (wstrb_hold[0] && wdata_hold[0]) begin
                            if (command_pending || program_active) begin
                                wrapper_error <= 1'b1;
                            end else begin
                                submitted_command <= {command_register[3], command_register[2],
                                                      command_register[1], command_register[0]};
                                command_pending <= 1'b1;
                                done_sticky <= 1'b0;
                            end
                        end
                        if (wstrb_hold[0] && wdata_hold[2]) begin
                            if (command_pending || program_active) begin
                                wrapper_error <= 1'b1;
                            end else begin
                                program_start_r <= 1'b1;
                                done_sticky <= 1'b0;
                            end
                        end
                    end
                    ADDR_COMMAND0: command_register[0] <= apply_write_strobes(
                        command_register[0], wdata_hold, wstrb_hold);
                    ADDR_COMMAND1: command_register[1] <= apply_write_strobes(
                        command_register[1], wdata_hold, wstrb_hold);
                    ADDR_COMMAND2: command_register[2] <= apply_write_strobes(
                        command_register[2], wdata_hold, wstrb_hold);
                    ADDR_COMMAND3: command_register[3] <= apply_write_strobes(
                        command_register[3], wdata_hold, wstrb_hold);
                    ADDR_PROGRAM_OUTPUT_BEATS: program_output_beats_r
                        <= apply_write_strobes(
                            program_output_beats_r, wdata_hold, wstrb_hold);
                    default: wrapper_error <= 1'b1;
                endcase
                aw_hold <= 1'b0;
                w_hold <= 1'b0;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rdata <= register_read(s_axi_araddr[7:0]);
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    initial begin
        if (AXI_ADDR_BITS < 8) $fatal(1, "AXI_ADDR_BITS must be at least 8");
    end
endmodule

`default_nettype wire
