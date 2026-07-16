if {$argc != 5} {
    error "Usage: annotate_saif_power.tcl <routed.dcp> <wrapper.saif> <core.saif> <output_dir> <accelerator_instance|__ROOT__>"
}
set checkpoint [file normalize [lindex $argv 0]]
set wrapper_saif [file normalize [lindex $argv 1]]
set core_saif [file normalize [lindex $argv 2]]
set output_dir [file normalize [lindex $argv 3]]
set accelerator_instance [lindex $argv 4]
file mkdir $output_dir
foreach required_file [list $checkpoint $wrapper_saif $core_saif] {
    if {![file exists $required_file]} {
        error "Required power-analysis input does not exist: $required_file"
    }
}

proc relocate_saif {source_path destination_path target_path} {
    set target_parts [split $target_path "/"]
    if {[llength $target_parts] < 2} {
        error "Relocated SAIF target must contain at least two hierarchy levels"
    }
    set source_handle [open $source_path "r"]
    set destination_handle [open $destination_path "w"]
    set replaced_instances 0
    try {
        while {[gets $source_handle line] >= 0} {
            if {$replaced_instances < 2 &&
                    [regexp {^(\s*)\(INSTANCE\s+\S+\s*$} $line -> indent]} {
                puts $destination_handle "${indent}(INSTANCE  [lindex $target_parts $replaced_instances]"
                incr replaced_instances
                if {$replaced_instances == 2} {
                    for {set index 2} {$index < [llength $target_parts]} {incr index} {
                        append indent "   "
                        puts $destination_handle "${indent}(INSTANCE  [lindex $target_parts $index]"
                    }
                }
            } else {
                puts $destination_handle $line
            }
        }
        if {$replaced_instances != 2} {
            error "Could not find the two simulated SAIF wrapper instances in $source_path"
        }
        for {set index 2} {$index < [llength $target_parts]} {incr index} {
            puts $destination_handle ")"
        }
    } finally {
        close $source_handle
        close $destination_handle
    }
}

open_checkpoint $checkpoint
if {$accelerator_instance eq "__ROOT__"} {
    read_saif -out_file [file join $output_dir "wrapper_saif_match.rpt"] \
        -strip_path tb_neuraedge_pynq_axi_top/dut -file $wrapper_saif
    current_instance u_core
    read_saif -out_file [file join $output_dir "core_saif_match.rpt"] \
        -strip_path tb_neuraedge_universal_core/dut -file $core_saif
    current_instance
} else {
    set accelerator_cells [get_cells -quiet -hierarchical -filter \
        "NAME == $accelerator_instance"]
    if {[llength $accelerator_cells] != 1} {
        error "Accelerator instance does not resolve uniquely: $accelerator_instance"
    }
    set relocated_wrapper [file join $output_dir "relocated_wrapper.saif"]
    set relocated_core [file join $output_dir "relocated_core.saif"]
    relocate_saif $wrapper_saif $relocated_wrapper $accelerator_instance
    relocate_saif $core_saif $relocated_core ${accelerator_instance}/u_core
    try {
        read_saif -out_file [file join $output_dir "wrapper_saif_match.rpt"] \
            -no_strip -file $relocated_wrapper
        read_saif -out_file [file join $output_dir "core_saif_match.rpt"] \
            -no_strip -file $relocated_core
    } finally {
        file delete -force $relocated_wrapper $relocated_core
    }
}
set_switching_activity -deassert_resets
if {$accelerator_instance eq "__ROOT__"} {
    set routed_nets [get_nets -hierarchical]
} else {
    current_instance $accelerator_instance
    set routed_nets [get_nets -hierarchical]
    current_instance
}
if {[llength $routed_nets] == 0} {
    error "The routed checkpoint contains no hierarchical nets to report"
}
report_switching_activity -file \
    [file join $output_dir "annotated_switching_activity.rpt"] $routed_nets
report_power -file [file join $output_dir "routed_saif_power.rpt"]
write_checkpoint -force [file join $output_dir "neuraedge_saif_annotated.dcp"]
close_design
