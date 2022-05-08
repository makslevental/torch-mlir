create_project project_1 project_1 -part xcu280-fsvh2892-2L-e -force
set_property board_part xilinx.com:au280:part0:1.1 [current_project]

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc

add_files -norecurse -scan_for_includes delayline_bram_514.vhd -force
import_files -norecurse delayline_bram_514.vhd -force

add_files -norecurse -scan_for_includes bramsr.v -force
import_files -norecurse bramsr.v -force

update_compile_order -fileset sources_1
set_property top bramsr [current_fileset]

set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING on [get_runs synth_1]

launch_runs synth_1 -jobs 16
wait_on_run synth_1
open_run synth_1
#opt_design
launch_runs impl_1 -jobs 16
report_utilization -file project_1/hier_util.rpt -hierarchical -hierarchical_percentages
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports ap_clk]
report_timing -file project_1/timing_synth.rpt



