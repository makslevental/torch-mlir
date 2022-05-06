create_project braggnn_vivado vivado -part xcu280-fsvh2892-2L-e -force
set_property board_part xilinx.com:au280:part0:1.1 [current_project]
create_clock -name SysClk -period 10 -waveform {0 5} [get_ports clk]

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc

add_files -scan_for_includes {forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
import_files {forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
source forward_fmuladd_32ns_32ns_32_10_med_dsp_1_ip.tcl

update_compile_order -fileset sources_1

add_files -norecurse -scan_for_includes forward.v -force
import_files -norecurse forward.v -force

set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING on [get_runs synth_1]

launch_runs synth_1 -jobs 16
wait_on_run synth_1
open_run synth_1
opt_design
report_utilization -file hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -file timing_synth.log


foreach IPfile [get_ips] {
    puts "\n $IPfile: \n";
    foreach prop [list_property [get_ips $IPfile] -regexp {^CONFIG\.\w+$}] {
        puts "$prop \{[get_property $prop [get_ips $IPfile]]\}";
}}