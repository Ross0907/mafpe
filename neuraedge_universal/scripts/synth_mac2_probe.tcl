set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".."]]
set out_dir [file join $project_dir "outputs" "mac2_probe"]
file mkdir $out_dir

read_verilog -sv [file join $project_dir "rtl" "neuraedge_mac2_u8.sv"]
read_verilog -sv [file join $project_dir "rtl" "neuraedge_mac2_array_probe.sv"]
synth_design -top neuraedge_mac2_array_probe -part xc7z020clg400-1 -generic PAIRS=128
create_clock -period 5.0 -name clk [get_ports clk]

opt_design
place_design
route_design

report_utilization -file [file join $out_dir "mac2_probe_utilization.rpt"]
report_timing_summary -file [file join $out_dir "mac2_probe_timing.rpt"]
report_drc -file [file join $out_dir "mac2_probe_drc.rpt"]
write_checkpoint -force [file join $out_dir "mac2_probe_routed.dcp"]

set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME =~ DSP48*}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fh [open [file join $out_dir "mac2_probe_summary.txt"] "w"]
puts $fh "pairs=128"
puts $fh "products_per_cycle=256"
puts $fh "dsp48_count=$dsp_count"
puts $fh "wns_ns=$wns"
close $fh

if {$dsp_count != 128} {
    error "Packed MAC probe expected exactly 128 DSP48 cells, got $dsp_count"
}
if {$wns < 0.0} {
    error "Packed MAC probe failed the 5.0 ns timing target: WNS=$wns"
}
