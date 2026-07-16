# Report module-level area and control-set pressure from any synthesized DCP.
if {$argc != 2} {
    error "Usage: report_checkpoint_hierarchy.tcl <checkpoint.dcp> <output_dir>"
}

set checkpoint [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
if {![file exists $checkpoint]} {
    error "Checkpoint does not exist: $checkpoint"
}
file mkdir $output_dir

open_checkpoint $checkpoint
report_utilization -hierarchical -hierarchical_depth 8 \
    -file [file join $output_dir "hierarchical_utilization.rpt"]
report_control_sets -verbose \
    -file [file join $output_dir "control_sets.rpt"]
report_design_analysis -complexity -congestion \
    -file [file join $output_dir "design_complexity_congestion.rpt"]
puts "NEURAEDGE_CHECKPOINT_HIERARCHY_REPORTED output=$output_dir"
close_design
exit 0
