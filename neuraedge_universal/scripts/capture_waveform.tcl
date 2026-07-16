if {![info exists ::env(NEURAEDGE_VCD_PATH)]} {
    error "NEURAEDGE_VCD_PATH is required"
}
if {![info exists ::env(NEURAEDGE_WAVE_AMOUNT)]} {
    error "NEURAEDGE_WAVE_AMOUNT is required"
}
if {![info exists ::env(NEURAEDGE_WAVE_UNIT)]} {
    error "NEURAEDGE_WAVE_UNIT is required"
}

set vcd_path [file normalize $::env(NEURAEDGE_VCD_PATH)]
set run_amount $::env(NEURAEDGE_WAVE_AMOUNT)
set run_unit $::env(NEURAEDGE_WAVE_UNIT)

open_vcd $vcd_path
log_vcd [get_objects -r /*]
log_wave -r /*
run $run_amount $run_unit
close_vcd
quit
