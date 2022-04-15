create_project braggnn_vivado vivado -part xcu280-fsvh2892-2L-e
set_property board_part xilinx.com:au280:part0:1.1 [current_project]

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc
add_files -scan_for_includes {forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
import_files {forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
source forward_fmuladd_32ns_32ns_32_10_med_dsp_1_ip.tcl
update_compile_order -fileset sources_1

add_files -norecurse -scan_for_includes forward.v -force
import_files -norecurse forward.v -force

update_compile_order -fileset sources_1
reset_run synth_1
launch_runs synth_1 -jobs 16