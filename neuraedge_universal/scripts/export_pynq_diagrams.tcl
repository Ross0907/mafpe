if {$argc != 2} {
    error "Usage: export_pynq_diagrams.tcl <project.xpr> <output_dir>"
}

set project_file [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
if {![file exists $project_file]} {
    error "Vivado project does not exist: $project_file"
}
file mkdir $output_dir

open_project $project_file
set bd_files [get_files -quiet system.bd]
if {[llength $bd_files] != 1} {
    error "Expected exactly one system.bd, found [llength $bd_files]"
}
open_bd_design [lindex $bd_files 0]

set pdf_path [file join $output_dir "neuraedge_pynq_block_design.pdf"]
set svg_path [file join $output_dir "neuraedge_pynq_block_design.svg"]
write_bd_layout -force -format pdf -orientation landscape -scope all $pdf_path
write_bd_layout -force -format svg -orientation landscape -scope all $svg_path
foreach diagram [list $pdf_path $svg_path] {
    if {![file exists $diagram] || [file size $diagram] == 0} {
        error "Block-design diagram was not generated: $diagram"
    }
}

puts "NEURAEDGE_PYNQ_DIAGRAMS_EXPORTED pdf=$pdf_path svg=$svg_path"
close_project
exit 0
