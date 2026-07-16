foreach required_variable {
    NEURAEDGE_SAIF_PATH
    NEURAEDGE_SAIF_SCOPE
    NEURAEDGE_SAIF_AMOUNT
    NEURAEDGE_SAIF_UNIT
} {
    if {![info exists ::env($required_variable)]} {
        error "$required_variable is required"
    }
}

set saif_path [file normalize $::env(NEURAEDGE_SAIF_PATH)]
set saif_scope $::env(NEURAEDGE_SAIF_SCOPE)
set run_amount $::env(NEURAEDGE_SAIF_AMOUNT)
set run_unit $::env(NEURAEDGE_SAIF_UNIT)
set activity_objects [get_objects -r "${saif_scope}/*"]
if {[llength $activity_objects] == 0} {
    error "No simulation objects matched SAIF scope $saif_scope"
}

open_saif $saif_path
log_saif $activity_objects
run $run_amount $run_unit
close_saif
quit
