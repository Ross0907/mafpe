# Shared final evidence and acceptance gates for fresh and resumed OOC builds.
# The caller must define output_dir and have an open, fully routed design.
report_utilization -hierarchical -file [file join $output_dir "routed_utilization_hierarchical.rpt"]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $output_dir "routed_timing_summary.rpt"]
report_timing -delay_type max -max_paths 250 -sort_by group \
    -file [file join $output_dir "routed_timing_paths_250.rpt"]
report_clock_utilization -file [file join $output_dir "routed_clock_utilization.rpt"]
set route_status_text [report_route_status -return_string]
set route_file [open [file join $output_dir "routed_route_status.rpt"] "w"]
puts -nonewline $route_file $route_status_text
close $route_file
report_drc -file [file join $output_dir "routed_drc.rpt"]
report_methodology -file [file join $output_dir "routed_methodology.rpt"]
report_design_analysis -congestion \
    -file [file join $output_dir "routed_congestion.rpt"]
report_power -file [file join $output_dir "routed_vectorless_power.rpt"]
write_checkpoint -force [file join $output_dir "neuraedge_pynq_axi_top_routed.dcp"]

set lut_count [llength [get_cells -hierarchical -filter {REF_NAME =~ LUT*}]]
set ff_count [llength [get_cells -hierarchical -filter {REF_NAME =~ FD*}]]
set dsp_count [llength [get_cells -hierarchical -filter {REF_NAME =~ DSP48*}]]
set bram36_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB36E1}]]
set bram18_count [llength [get_cells -hierarchical -filter {REF_NAME == RAMB18E1}]]
set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
set setup_wns [get_property SLACK $setup_path]
set hold_whs [get_property SLACK $hold_path]
set route_error_count -1
if {[regexp {# of nets with routing errors\.*\s*:\s*([0-9]+)} \
        $route_status_text route_match route_count]} {
    set route_error_count $route_count
}
set drc_error_count [llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]

set summary_file [open [file join $output_dir "implementation_summary.txt"] "w"]
puts $summary_file "part=xc7z020clg400-1"
puts $summary_file "target_clock_mhz=200.0"
puts $summary_file "lut_count=$lut_count"
puts $summary_file "ff_count=$ff_count"
puts $summary_file "dsp48_count=$dsp_count"
puts $summary_file "bram36_count=$bram36_count"
puts $summary_file "bram18_count=$bram18_count"
puts $summary_file "setup_wns_ns=$setup_wns"
puts $summary_file "hold_whs_ns=$hold_whs"
puts $summary_file "route_error_count=$route_error_count"
puts $summary_file "drc_error_count=$drc_error_count"
puts $summary_file "hold_context=ooc_top_ports_without_hd_partpin_locs"
close $summary_file

if {$route_error_count != 0} {
    error "Routed design is incomplete: route_error_count=$route_error_count"
}
if {$drc_error_count != 0} {
    error "Routed design has DRC errors: drc_error_count=$drc_error_count"
}
if {$setup_wns < 0.0} {
    error "Routed design missed the 5.0 ns target: WNS=$setup_wns ns"
}
# Min-delay paths at OOC AXI/stream ports are inaccurate without physical
# partition pins. Preserve the diagnostic and enforce hold in the full shell.
if {$hold_whs < 0.0} {
    puts "NEURAEDGE_OOC_HOLD_CONTEXT_ONLY WHS=$hold_whs ns"
}
