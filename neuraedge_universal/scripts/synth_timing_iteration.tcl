set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".."]]
set output_name "timing_iteration"
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

report_utilization -hierarchical \
    -file [file join $output_dir "post_synth_utilization_hierarchical.rpt"]
report_timing -delay_type max -max_paths 250 -nworst 1 -input_pins \
    -file [file join $output_dir "post_synth_timing_paths_250.rpt"]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $output_dir "post_synth_timing_summary.rpt"]
write_checkpoint -force [file join $output_dir "post_synth.dcp"]
