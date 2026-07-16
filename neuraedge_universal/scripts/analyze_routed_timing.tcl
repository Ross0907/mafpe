set script_dir [file dirname [file normalize [info script]]]
set project_dir [file dirname $script_dir]
set output_dir [file join $project_dir "outputs" "pynq_ooc"]

open_checkpoint [file join $output_dir "neuraedge_pynq_axi_top_routed.dcp"]
report_timing -delay_type max -max_paths 250 -nworst 1 -input_pins \
    -file [file join $output_dir "routed_timing_paths_250.rpt"]
report_methodology -file [file join $output_dir "routed_methodology.rpt"]
close_design
