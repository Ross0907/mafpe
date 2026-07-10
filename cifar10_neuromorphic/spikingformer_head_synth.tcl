set project_dir [file normalize [file dirname [info script]]]
set repo_root [file normalize [file join $project_dir ".."]]

if {![info exists ::env(NEURAEDGE_PART)]} {
    set part_name "xc7z020clg400-1"
} else {
    set part_name $::env(NEURAEDGE_PART)
}

if {![info exists ::env(NEURAEDGE_CLOCK_NS)]} {
    set clock_ns 10.0
} else {
    set clock_ns $::env(NEURAEDGE_CLOCK_NS)
}

if {![info exists ::env(NEURAEDGE_OUT_DIR)]} {
    set out_dir [file join $project_dir "outputs_spikingformer_head_accelerator" "vivado"]
} else {
    set out_dir [file normalize $::env(NEURAEDGE_OUT_DIR)]
}

file mkdir $out_dir
cd $project_dir

read_verilog -sv [file join $project_dir "rtl" "cifar10_spikingformer_head_accelerator.sv"]
read_verilog -sv [file join $project_dir "rtl" "cifar10_spikingformer_head_pynq_axi_stream_top.sv"]
synth_design -top cifar10_spikingformer_head_pynq_axi_stream_top -part $part_name
create_clock -period $clock_ns -name ap_clk [get_ports ap_clk]

opt_design
place_design
route_design

report_utilization -file [file join $out_dir "spikingformer_head_utilization.rpt"]
report_timing_summary -file [file join $out_dir "spikingformer_head_timing_summary.rpt"]
report_power -file [file join $out_dir "spikingformer_head_power.rpt"]
write_checkpoint -force [file join $out_dir "spikingformer_head_routed.dcp"]

set timing_ok [expr {[get_property SLACK [get_timing_paths -max_paths 1 -setup]] >= 0.0}]
set fh [open [file join $out_dir "spikingformer_head_synth_summary.txt"] "w"]
puts $fh "top=cifar10_spikingformer_head_pynq_axi_stream_top"
puts $fh "part=$part_name"
puts $fh "clock_ns=$clock_ns"
puts $fh "timing_met=$timing_ok"
close $fh
