set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".."]]
set output_name "pynq_ooc"
if {$argc >= 1} {
    set output_name [lindex $argv 0]
}
set output_dir [file join $project_dir "outputs" $output_name]
file mkdir $output_dir

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

read_verilog -sv $rtl_files
synth_design -top neuraedge_pynq_axi_top -part xc7z020clg400-1 \
    -mode out_of_context -flatten_hierarchy rebuilt -directive PerformanceOptimized
create_clock -name aclk -period 5.000 [get_ports aclk]
set_clock_uncertainty 0.100 [get_clocks aclk]
set_property HD.CLK_SRC BUFGCTRL_X0Y0 [get_ports aclk]
set_false_path -from [get_ports aresetn]
set data_inputs [get_ports -filter {DIRECTION == IN && NAME != "aclk" && NAME != "aresetn"}]
set data_outputs [get_ports -filter {DIRECTION == OUT}]
set_input_delay -clock aclk 1.000 $data_inputs
set_output_delay -clock aclk 1.000 $data_outputs

report_utilization -hierarchical -file [file join $output_dir "post_synth_utilization_hierarchical.rpt"]
report_timing_summary -file [file join $output_dir "post_synth_timing.rpt"]
write_checkpoint -force [file join $output_dir "post_synth.dcp"]

opt_design -directive Explore
place_design -directive ExtraNetDelay_high
phys_opt_design -directive AggressiveExplore
route_design -directive AggressiveExplore
# Persist the expensive placed-and-routed solution before post-route physical
# synthesis. Vivado can then resume without repeating placement if its
# optimizer process terminates unexpectedly.
write_checkpoint -force [file join $output_dir "pre_postroute_physopt.dcp"]
# Vivado 2026.1 on Windows reproducibly faults in AggressiveExplore's
# post-route hold-fix phase. Explore performs the same critical-path repair
# without entering that faulty phase; aggressive optimization remains enabled
# before routing, where it completes normally.
phys_opt_design -directive Explore
write_checkpoint -force [file join $output_dir "postroute_physopt_unrouted.dcp"]
# Post-route physical optimization may move cells or insert hold buffers. Route
# those edits before reporting timing so setup and hold are both physically real.
route_design -directive Quick
source [file join $script_dir "finalize_pynq_ooc.tcl"]
