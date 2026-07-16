set script_dir [file normalize [file dirname [info script]]]
set project_dir [file normalize [file join $script_dir ".."]]
set output_name "pynq_ooc"
set directive "Explore"
if {$argc >= 1} {
    set output_name [lindex $argv 0]
}
if {$argc >= 2} {
    set directive [lindex $argv 1]
}
set output_dir [file join $project_dir "outputs" $output_name]
set checkpoint [file join $output_dir "pre_postroute_physopt.dcp"]

if {![file exists $checkpoint]} {
    error "Routed resume checkpoint does not exist: $checkpoint"
}

open_checkpoint $checkpoint
phys_opt_design -directive $directive
write_checkpoint -force [file join $output_dir "postroute_physopt_unrouted.dcp"]
route_design -directive Quick
source [file join $script_dir "finalize_pynq_ooc.tcl"]
