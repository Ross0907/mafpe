# Inspect the implemented optimizer multipliers without modifying the checkpoint.
if {$argc != 2} {
    error "Usage: report_optimizer_dsp.tcl <checkpoint.dcp> <output_report>"
}

set checkpoint [file normalize [lindex $argv 0]]
set output_report [file normalize [lindex $argv 1]]
if {![file exists $checkpoint]} {
    error "Checkpoint does not exist: $checkpoint"
}
file mkdir [file dirname $output_report]

open_checkpoint $checkpoint
set optimizer_dsps [get_cells -hierarchical -filter {
    REF_NAME == DSP48E1 && NAME =~ *u_optimizer*
}]

set report_handle [open $output_report w]
puts $report_handle "checkpoint=$checkpoint"
puts $report_handle "optimizer_dsp_count=[llength $optimizer_dsps]"
foreach dsp $optimizer_dsps {
    puts $report_handle "\n=== [get_property NAME $dsp] ==="
    foreach property [lsort [list_property $dsp]] {
        puts $report_handle "$property=[get_property $property $dsp]"
    }
    foreach pin_name {A B C D M P CLK CEA1 CEA2 CEB1 CEB2 CEM CEP RSTA RSTB RSTM RSTP} {
        set pins [get_pins -quiet "$dsp/$pin_name*"]
        if {[llength $pins] == 0} {
            continue
        }
        foreach pin $pins {
            set nets [get_nets -quiet -of_objects $pin]
            if {[llength $nets] > 0} {
                puts $report_handle \
                    "PIN [get_property NAME $pin] NET [join [get_property NAME $nets] {, }]"
            }
        }
    }
}
close $report_handle

puts "NEURAEDGE_OPTIMIZER_DSP_REPORTED count=[llength $optimizer_dsps] report=$output_report"
close_design
exit 0
