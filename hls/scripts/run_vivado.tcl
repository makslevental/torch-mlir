# BEGIN Vivado Commands
set vivado_ver [version -short]
set fpo_ver 7.1
if {[regexp -nocase {2015\.1.*} \
 $vivado_ver match]} \
 {
    set fpo_ver 7.0
} \

create_project project_1 project_1 -part xcu280-fsvh2892-2L-e -force
set_property board_part xilinx.com:au280:part0:1.1 [current_project]

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc

add_files -norecurse -scan_for_includes forward.v -force
import_files -norecurse forward.v -force

add_files -norecurse -scan_for_includes relu.v -force
import_files -norecurse relu.v -force

add_files -norecurse -scan_for_includes neg.v -force
import_files -norecurse neg.v -force

add_files -norecurse -scan_for_includes rom.v -force
import_files -norecurse rom.v -force

add_files -norecurse -scan_for_includes simple_dual_rw_ram.v -force
import_files -norecurse simple_dual_rw_ram.v -force

source half_fadd.tcl
source half_fmul.tcl

update_compile_order -fileset sources_1

set_property top forward [current_fileset]
set_property top forward [current_fileset]
set_property strategy Flow_AlternateRoutability [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING off [get_runs synth_1]
#set_property strategy Congestion_SSI_SpreadLogic_high [get_runs impl_1]
set_property strategy Performance_WLBlockPlacementFanoutOpt [get_runs impl_1]

launch_runs synth_1 -jobs 16
wait_on_run synth_1
open_run synth_1

create_clock -name clk -period 10 -waveform {0 5} [get_ports clock]

file mkdir reports
report_utilization -file reports/syn_before_opt_util.rpt
report_utilization -file reports/syn_before_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -file reports/syn_before_opt_timing.rpt -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins
report_power -file reports/syn_before_opt_power.rpt

opt_design

report_utilization -file reports/syn_after_opt_util.rpt
report_utilization -file reports/syn_after_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -file reports/syn_after_opt_timing.rpt -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins
report_power -file reports/syn_after_opt_power.rpt

launch_runs impl_1 -jobs 16
wait_on_run impl_1
open_run impl_1

report_utilization -file reports/impl_after_opt_util.rpt
report_utilization -file reports/impl_after_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -file reports/impl_after_opt_timing.rpt -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins
report_power -file reports/impl_after_opt_power.rpt