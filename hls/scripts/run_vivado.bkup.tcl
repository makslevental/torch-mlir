create_project braggnn_vivado braggnn_vivado -part xcu280-fsvh2892-2L-e -force
create_project braggnn_vivado braggnn_vivado -part xcu280-fsvh2892-2L-e -force
set_property board_part xilinx.com:au280:part0:1.1 [current_project]

add_files -norecurse -scan_for_includes /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/forward.1.v -force
import_files -norecurse /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/forward.1.v -force

add_files -fileset constrs_1 -norecurse /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/alveo-u280-xdc.xdc
import_files -fileset constrs_1 /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/alveo-u280-xdc.xdc

#add_files -scan_for_includes {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fadd_32ns_32ns_32_13_full_dsp_1.v}
#import_files {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fadd_32ns_32ns_32_13_full_dsp_1.v}
#source /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fadd_32ns_32ns_32_13_full_dsp_1_ip.tcl
#
#add_files -scan_for_includes {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fmul_32ns_32ns_32_10_med_dsp_1.v}
#import_files {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fmul_32ns_32ns_32_10_med_dsp_1.v}
#source /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/proj/forward_fmul_32ns_32ns_32_10_med_dsp_1_ip.tcl

add_files -scan_for_includes {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
import_files {/home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/forward_fmuladd_32ns_32ns_32_10_med_dsp_1.v}
source /home/mlevental/dev_projects/torch-mlir/hls/examples/cnn_layers_1.1/forward_fmuladd_32ns_32ns_32_10_med_dsp_1_ip.tcl

update_compile_order -fileset sources_1

reset_run synth_1
#set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
#set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AlternateRoutability [get_runs synth_1]
#set_property strategy Flow_AreaOptimized_High [get_runs synth_1]

set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RESOURCE_SHARING on [get_runs synth_1]
#set_property strategy Performance_ExploreWithRemap [get_runs synth_1]

launch_runs synth_1 -jobs 16
wait_on_run synth_1
open_run synth_1
opt_design
report_utilization -file hier_util.rpt -hierarchical -hierarchical_percentages
report_timing_summary -file timing_synth.log