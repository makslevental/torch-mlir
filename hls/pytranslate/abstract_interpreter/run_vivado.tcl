#open_project /home/mlevental/dev_projects/vivado_projects/braggnn/braggnn.xpr
set_property PART "xcau25p-sfvb784-1LV-i" [current_project]
add_files -norecurse -scan_for_includes /home/mlevental/dev_projects/torch-mlir/hls/pytranslate/abstract_interpreter/schedule.v -force
import_files -norecurse /home/mlevental/dev_projects/torch-mlir/hls/pytranslate/abstract_interpreter/schedule.v -force
reset_run synth_1
launch_runs synth_1 -jobs 16