create_project project_1 project_1 -part xcu280-fsvh2892-2L-e -force
set_property board_part xilinx.com:au280:part0:1.1 [current_project]

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc

add_files -norecurse -scan_for_includes delayline_bram_514.vhd -force
import_files -norecurse delayline_bram_514.vhd -force
add_files -norecurse -scan_for_includes bramsr.v -force
import_files -norecurse bramsr.v -force

update_compile_order -fileset sources_1

set_property top forward [current_fileset]
set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING on [get_runs synth_1]

launch_runs synth_1 -jobs 16
wait_on_run synth_1
open_run synth_1

report_utilization -file reports/syn_before_opt_util.rpt
report_utilization -file reports/syn_before_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -file reports/syn_before_opt_timing.rpt
report_power -file reports/syn_before_opt_power.rpt

opt_design

report_utilization -file reports/syn_after_opt_util.rpt
report_utilization -file reports/syn_after_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -file reports/syn_after_opt_timing.rpt
report_power -file reports/syn_after_opt_power.rpt

launch_runs impl_1 -jobs 16
wait_on_run impl_1
open_run impl_1

report_utilization -file reports/impl_after_opt_util.rpt
report_utilization -file reports/impl_after_opt_hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -delay_type max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -file reports/impl_after_opt_timing.rpt
report_power -file reports/impl_after_opt_power.rpt
