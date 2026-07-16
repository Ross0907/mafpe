# Build the complete PYNQ-Z2 processing-system, DMA, and NeuraEdge overlay.
# The board preset is vendored from the AMD/Xilinx Board Store and configures
# the physical DDR3 and MIO pins. FCLK0 supplies 100 MHz control, FCLK1 supplies
# a timing-safe 125 MHz DMA/DDR path, and FCLK2 supplies the 200 MHz accelerator.

set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".."]]
set board_store [file join $project_dir "third_party" "XilinxBoardStore" "boards"]
set output_dir [file join $project_dir "outputs" "pynq_full"]
if {[info exists ::env(NEURAEDGE_VIVADO_BUILD_ROOT)]} {
    set build_root [file normalize $::env(NEURAEDGE_VIVADO_BUILD_ROOT)]
} else {
    set build_root [file normalize "D:/NeuraEdgeBuild"]
}
set vivado_project_dir [file join $build_root "pynq_full"]
set build_mode "full"
if {$argc >= 1} {
    set build_mode [lindex $argv 0]
}
if {$build_mode ni {validate full}} {
    error "Build mode must be validate or full"
}
file mkdir $output_dir
if {$build_mode eq "full"} {
    # A failed rebuild must never leave an older deployable image looking like
    # the result of the current source tree. Diagnostic reports are overwritten
    # in place, while accepted deployment artifacts are republished only after
    # route, timing, and DRC gates pass below.
    foreach stale_artifact {
        neuraedge_pynq_full.bit
        neuraedge_pynq_full.hwh
        neuraedge_pynq_full.xsa
        full_implementation_summary.txt
    } {
        set stale_path [file join $output_dir $stale_artifact]
        if {[file exists $stale_path]} {
            file delete -force $stale_path
        }
    }
}

set_param board.repoPaths [list $board_store]
set board_part "tul.com.tw:pynq-z2:part0:1.0"
if {[llength [get_board_parts -quiet $board_part]] != 1} {
    error "Vendored PYNQ-Z2 board part was not discovered at $board_store"
}

create_project neuraedge_pynq_full $vivado_project_dir -force -part xc7z020clg400-1
set_property board_part $board_part [current_project]
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set rtl_files [list \
    [file join $project_dir "rtl" "neuraedge_command_pkg.sv"] \
    [file join $project_dir "rtl" "neuraedge_mac2_u8.sv"] \
    [file join $project_dir "rtl" "neuraedge_tensor_tile.sv"] \
    [file join $project_dir "rtl" "neuraedge_active_index_buffer.sv"] \
    [file join $project_dir "rtl" "neuraedge_activation_buffer.sv"] \
    [file join $project_dir "rtl" "neuraedge_weight_bank.sv"] \
    [file join $project_dir "rtl" "neuraedge_buffered_tensor_core.sv"] \
    [file join $project_dir "rtl" "neuraedge_alpha_xnor.sv"] \
    [file join $project_dir "rtl" "neuraedge_vector_alu.sv"] \
    [file join $project_dir "rtl" "neuraedge_udivider.sv"] \
    [file join $project_dir "rtl" "neuraedge_isqrt64.sv"] \
    [file join $project_dir "rtl" "neuraedge_integer_layernorm.sv"] \
    [file join $project_dir "rtl" "neuraedge_log_softmax.sv"] \
    [file join $project_dir "rtl" "neuraedge_lif_bank.sv"] \
    [file join $project_dir "rtl" "neuraedge_sgd_optimizer8.sv"] \
    [file join $project_dir "rtl" "neuraedge_gradient_accumulator_bank.sv"] \
    [file join $project_dir "rtl" "neuraedge_linear_backward8.sv"] \
    [file join $project_dir "rtl" "neuraedge_activation_lut8.sv"] \
    [file join $project_dir "rtl" "neuraedge_universal_core.sv"] \
    [file join $project_dir "rtl" "neuraedge_descriptor_sequencer.sv"] \
    [file join $project_dir "rtl" "neuraedge_pynq_axi_top.sv"]]
add_files -norecurse $rtl_files
set_property file_type SystemVerilog [get_files $rtl_files]
set_property top neuraedge_pynq_axi_top [current_fileset]
update_compile_order -fileset sources_1

# Package the verified RTL with an explicit IP-XACT memory map. A raw module
# reference exposes AXI pins but has no address block, which prevents Vivado's
# Address Editor and PYNQ's HWH parser from describing the control registers.
set ip_repo [file join $output_dir "ip_repo" "neuraedge_universal_1_0"]
file mkdir $ip_repo
ipx::package_project -root_dir $ip_repo -vendor neuraedge.local \
    -library accelerator -taxonomy /UserIP -import_files -set_current true
set packaged_core [ipx::current_core]
set_property name neuraedge_universal $packaged_core
set_property display_name "NeuraEdge Universal Accelerator" $packaged_core
set_property description "INT8, sparse, spiking, transformer, and on-fabric optimizer accelerator" $packaged_core
set_property version 1.0 $packaged_core
set_property core_revision 1 $packaged_core
set s_axi_interface [ipx::get_bus_interfaces S_AXI -of_objects $packaged_core]
ipx::add_memory_map S_AXI $packaged_core
set_property slave_memory_map_ref S_AXI $s_axi_interface
set memory_map [ipx::get_memory_maps S_AXI -of_objects $packaged_core]
ipx::add_address_block Reg $memory_map
set register_block [ipx::get_address_blocks Reg -of_objects $memory_map]
set_property range 256 $register_block
set_property width 32 $register_block
ipx::associate_bus_interfaces -busif S_AXI -clock aclk $packaged_core
ipx::associate_bus_interfaces -busif S_AXIS -clock aclk $packaged_core
ipx::associate_bus_interfaces -busif M_AXIS -clock aclk $packaged_core
ipx::create_xgui_files $packaged_core
ipx::update_checksums $packaged_core
ipx::save_core $packaged_core
set_property ip_repo_paths [list [file dirname $ip_repo]] [current_project]
update_ip_catalog

create_bd_design system

set ps7 [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {apply_board_preset "1" make_external "FIXED_IO, DDR" Master "Disable" Slave "Disable"} \
    $ps7
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_S_AXI_HP0_DATA_WIDTH {64} \
    CONFIG.PCW_S_AXI_HP0_ID_WIDTH {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_EN_CLK1_PORT {1} \
    CONFIG.PCW_EN_CLK2_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100.000000} \
    CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ {125.000000} \
    CONFIG.PCW_FPGA2_PERIPHERAL_FREQMHZ {200.000000}] $ps7
# The board preset assigns 10 MHz to this disabled-by-default interface. Once
# GP0 is enabled, let block-design propagation derive its frequency from the
# connected 100 MHz FCLK0 instead of retaining that stale user-strength value.
set_property CONFIG.PCW_M_AXI_GP0_FREQMHZ.VALUE_SRC auto $ps7

set reset_100 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_100]
set reset_125 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_125]
set reset_200 [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 reset_200]
set accelerator_control_cdc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 accelerator_control_cdc]
set dma_control_cdc [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 dma_control_cdc]
set control_interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 control_interconnect]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $control_interconnect

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
    CONFIG.c_m_axi_mm2s_data_width {64} \
    CONFIG.c_m_axi_s2mm_data_width {64} \
    CONFIG.c_m_axis_mm2s_tdata_width {64} \
    CONFIG.c_s_axis_s2mm_tdata_width {64} \
    CONFIG.c_mm2s_burst_size {256} \
    CONFIG.c_s2mm_burst_size {256}] $dma

set memory_interconnect [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 memory_interconnect]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $memory_interconnect

set mm2s_cdc [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 mm2s_cdc]
set s2mm_cdc [create_bd_cell -type ip -vlnv xilinx.com:ip:axis_clock_converter:1.1 s2mm_cdc]
foreach stream_cdc [list $mm2s_cdc $s2mm_cdc] {
    set_property -dict [list \
        CONFIG.TDATA_NUM_BYTES {8} \
        CONFIG.HAS_TKEEP {1} \
        CONFIG.HAS_TLAST {1} \
        CONFIG.IS_ACLK_ASYNC {1} \
        CONFIG.SYNCHRONIZATION_STAGES {3}] $stream_cdc
}

set accelerator [create_bd_cell -type ip \
    -vlnv neuraedge.local:accelerator:neuraedge_universal:1.0 accelerator]
set interrupt_concat [create_bd_cell -type inline_hdl \
    -vlnv xilinx.com:inline_hdl:ilconcat:1.0 interrupt_concat]
set_property CONFIG.NUM_PORTS {16} $interrupt_concat
set interrupt_zero [create_bd_cell -type inline_hdl \
    -vlnv xilinx.com:inline_hdl:ilconstant:1.0 interrupt_zero]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] $interrupt_zero

# Control path: the 100 MHz GP0/interconnect fans out through explicit CDCs to
# the 200 MHz accelerator and the 125 MHz DMA register interface.
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins control_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_interconnect/M00_AXI] [get_bd_intf_pins accelerator_control_cdc/S_AXI]
connect_bd_intf_net [get_bd_intf_pins accelerator_control_cdc/M_AXI] [get_bd_intf_pins accelerator/S_AXI]
connect_bd_intf_net [get_bd_intf_pins control_interconnect/M01_AXI] [get_bd_intf_pins dma_control_cdc/S_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_control_cdc/M_AXI] [get_bd_intf_pins dma/S_AXI_LITE]

# Streaming path: 125 MHz DDR/DMA -> asynchronous stream CDC -> 200 MHz
# accelerator -> asynchronous stream CDC -> 125 MHz DMA/DDR.
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins mm2s_cdc/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins mm2s_cdc/M_AXIS] [get_bd_intf_pins accelerator/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins accelerator/M_AXIS] [get_bd_intf_pins s2mm_cdc/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins s2mm_cdc/M_AXIS] [get_bd_intf_pins dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins memory_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins memory_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_interconnect/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]

# Clock and reset domains are explicit; there is no unconstrained interface CDC.
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
    [get_bd_pins ps7/M_AXI_GP0_ACLK] \
    [get_bd_pins reset_100/slowest_sync_clk] \
    [get_bd_pins control_interconnect/ACLK] \
    [get_bd_pins control_interconnect/S00_ACLK] \
    [get_bd_pins control_interconnect/M00_ACLK] \
    [get_bd_pins control_interconnect/M01_ACLK] \
    [get_bd_pins accelerator_control_cdc/s_axi_aclk] \
    [get_bd_pins dma_control_cdc/s_axi_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] \
    [get_bd_pins ps7/S_AXI_HP0_ACLK] \
    [get_bd_pins reset_125/slowest_sync_clk] \
    [get_bd_pins memory_interconnect/ACLK] \
    [get_bd_pins memory_interconnect/S00_ACLK] \
    [get_bd_pins memory_interconnect/S01_ACLK] \
    [get_bd_pins memory_interconnect/M00_ACLK] \
    [get_bd_pins dma/s_axi_lite_aclk] \
    [get_bd_pins dma/m_axi_mm2s_aclk] \
    [get_bd_pins dma/m_axi_s2mm_aclk] \
    [get_bd_pins dma_control_cdc/m_axi_aclk] \
    [get_bd_pins mm2s_cdc/s_axis_aclk] \
    [get_bd_pins s2mm_cdc/m_axis_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK2] \
    [get_bd_pins reset_200/slowest_sync_clk] \
    [get_bd_pins accelerator_control_cdc/m_axi_aclk] \
    [get_bd_pins mm2s_cdc/m_axis_aclk] \
    [get_bd_pins s2mm_cdc/s_axis_aclk] \
    [get_bd_pins accelerator/aclk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] \
    [get_bd_pins reset_100/ext_reset_in] \
    [get_bd_pins reset_125/ext_reset_in] \
    [get_bd_pins reset_200/ext_reset_in]
connect_bd_net [get_bd_pins reset_100/interconnect_aresetn] \
    [get_bd_pins control_interconnect/ARESETN] \
    [get_bd_pins control_interconnect/S00_ARESETN] \
    [get_bd_pins control_interconnect/M00_ARESETN] \
    [get_bd_pins control_interconnect/M01_ARESETN] \
    [get_bd_pins accelerator_control_cdc/s_axi_aresetn] \
    [get_bd_pins dma_control_cdc/s_axi_aresetn]
connect_bd_net [get_bd_pins reset_125/interconnect_aresetn] \
    [get_bd_pins dma_control_cdc/m_axi_aresetn] \
    [get_bd_pins memory_interconnect/ARESETN] \
    [get_bd_pins memory_interconnect/S00_ARESETN] \
    [get_bd_pins memory_interconnect/S01_ARESETN] \
    [get_bd_pins memory_interconnect/M00_ARESETN]
connect_bd_net [get_bd_pins reset_125/peripheral_aresetn] \
    [get_bd_pins dma/axi_resetn] \
    [get_bd_pins mm2s_cdc/s_axis_aresetn] \
    [get_bd_pins s2mm_cdc/m_axis_aresetn]
connect_bd_net [get_bd_pins reset_200/interconnect_aresetn] \
    [get_bd_pins accelerator_control_cdc/m_axi_aresetn]
connect_bd_net [get_bd_pins reset_200/peripheral_aresetn] \
    [get_bd_pins mm2s_cdc/m_axis_aresetn] \
    [get_bd_pins s2mm_cdc/s_axis_aresetn] \
    [get_bd_pins accelerator/aresetn]

# Accelerator completion and both DMA channels are independently visible to Linux.
connect_bd_net [get_bd_pins accelerator/interrupt] [get_bd_pins interrupt_concat/In0]
connect_bd_net [get_bd_pins dma/mm2s_introut] [get_bd_pins interrupt_concat/In1]
connect_bd_net [get_bd_pins dma/s2mm_introut] [get_bd_pins interrupt_concat/In2]
for {set interrupt_index 3} {$interrupt_index < 16} {incr interrupt_index} {
    connect_bd_net [get_bd_pins interrupt_zero/dout] \
        [get_bd_pins interrupt_concat/In$interrupt_index]
}
connect_bd_net [get_bd_pins interrupt_concat/dout] [get_bd_pins ps7/IRQ_F2P]

assign_bd_address -offset 0x43C00000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs accelerator/S_AXI/Reg] -force
assign_bd_address -offset 0x40400000 -range 0x00010000 \
    -target_address_space [get_bd_addr_spaces ps7/Data] \
    [get_bd_addr_segs dma/S_AXI_LITE/Reg] -force
assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces dma/Data_MM2S] \
    [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM] -force
assign_bd_address -offset 0x00000000 -range 0x20000000 \
    -target_address_space [get_bd_addr_spaces dma/Data_S2MM] \
    [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM] -force

validate_bd_design
save_bd_design
set bd_file [get_files system.bd]
generate_target all $bd_file
make_wrapper -files $bd_file -top
set wrapper_file [file join $vivado_project_dir "neuraedge_pynq_full.gen" "sources_1" "bd" "system" "hdl" "system_wrapper.v"]
add_files -norecurse $wrapper_file
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

write_bd_tcl -force [file join $output_dir "generated_system_bd.tcl"]
if {$build_mode eq "validate"} {
    report_ip_status -file [file join $output_dir "validated_ip_status.rpt"]
    puts "NEURAEDGE_PYNQ_BD_VALIDATION_PASSED"
    close_project
    exit 0
}

set synth_run [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE PerformanceOptimized $synth_run
set impl_run [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE Explore $impl_run
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraNetDelay_high $impl_run
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true $impl_run
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore $impl_run
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true $impl_run
# Vivado 2026.1 on Windows reproducibly faults in AggressiveExplore's
# post-route hold-fix phase for this design. Explore retains post-route setup
# and hold repair without entering that unstable phase.
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE Explore $impl_run

launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS $synth_run] ne "100%"
        || ![string match "*Complete*" [get_property STATUS $synth_run]]} {
    error "Full PYNQ synthesis failed: [get_property STATUS $synth_run]"
}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS $impl_run] ne "100%"
        || ![string match "*Complete*" [get_property STATUS $impl_run]]} {
    error "Full PYNQ implementation failed: [get_property STATUS $impl_run]"
}

open_run impl_1
report_utilization -hierarchical -file [file join $output_dir "full_routed_utilization_hierarchical.rpt"]
report_timing_summary -delay_type min_max -max_paths 50 \
    -file [file join $output_dir "full_routed_timing_summary.rpt"]
report_timing -delay_type max -max_paths 250 -sort_by group \
    -file [file join $output_dir "full_routed_timing_paths_250.rpt"]
report_clock_interaction -file [file join $output_dir "full_clock_interaction.rpt"]
report_cdc -details -file [file join $output_dir "full_cdc.rpt"]
set full_route_status_text [report_route_status -return_string]
set route_status_file [open [file join $output_dir "full_route_status.rpt"] "w"]
puts -nonewline $route_status_file $full_route_status_text
close $route_status_file
report_drc -file [file join $output_dir "full_drc.rpt"]
report_methodology -file [file join $output_dir "full_methodology.rpt"]
report_design_analysis -congestion -file [file join $output_dir "full_congestion.rpt"]
report_power -file [file join $output_dir "full_vectorless_power.rpt"]
write_checkpoint -force [file join $output_dir "neuraedge_pynq_full_routed.dcp"]

set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
set setup_wns [get_property SLACK $setup_path]
set hold_whs [get_property SLACK $hold_path]
set route_error_count -1
if {[regexp {# of nets with routing errors\.*\s*:\s*([0-9]+)} \
        $full_route_status_text route_match route_count]} {
    set route_error_count $route_count
}
set drc_error_count [llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]
set summary_file [open [file join $output_dir "full_implementation_summary.txt"] "w"]
puts $summary_file "board_part=$board_part"
puts $summary_file "part=xc7z020clg400-1"
puts $summary_file "accelerator_clock_mhz=200.0"
puts $summary_file "dma_clock_mhz=125.0"
puts $summary_file "control_clock_mhz=100.0"
puts $summary_file "lut_count=[llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]"
puts $summary_file "ff_count=[llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]"
puts $summary_file "dsp48_count=[llength [get_cells -hierarchical -filter {REF_NAME =~ DSP48*}]]"
puts $summary_file "bram36_count=[llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]"
puts $summary_file "bram18_count=[llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]"
puts $summary_file "setup_wns_ns=$setup_wns"
puts $summary_file "hold_whs_ns=$hold_whs"
puts $summary_file "route_error_count=$route_error_count"
puts $summary_file "drc_error_count=$drc_error_count"
close $summary_file

if {$route_error_count != 0} {
    error "Full PYNQ routing is incomplete: route_error_count=$route_error_count"
}
if {$drc_error_count != 0} {
    error "Full PYNQ design has DRC errors: drc_error_count=$drc_error_count"
}
if {$setup_wns < 0.0} {
    error "Full PYNQ design missed setup timing: WNS=$setup_wns ns"
}
if {$hold_whs < 0.0} {
    error "Full PYNQ design missed hold timing: WHS=$hold_whs ns"
}

set generated_bit [file join $vivado_project_dir \
    "neuraedge_pynq_full.runs" "impl_1" "system_wrapper.bit"]
if {![file exists $generated_bit]} {
    error "Implementation completed but bitstream was not found at $generated_bit"
}
file copy -force $generated_bit [file join $output_dir "neuraedge_pynq_full.bit"]
set generated_hwh [file join $vivado_project_dir "neuraedge_pynq_full.gen" "sources_1" "bd" "system" "hw_handoff" "system.hwh"]
if {![file exists $generated_hwh]} {
    error "Block design completed but hardware handoff was not found at $generated_hwh"
}
file copy -force $generated_hwh [file join $output_dir "neuraedge_pynq_full.hwh"]
write_hw_platform -fixed -include_bit -force \
    -file [file join $output_dir "neuraedge_pynq_full.xsa"]
