open_project -reset proj
add_files /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/wrapper.cpp
set_top wrapper

open_solution -reset solution1
#set_part "xcvu35p-fsvh2104-3-e"
set_part "xc7a200tfbg484-3"
create_clock -period "100MHz"
config_export -format ip_catalog -rtl verilog

set ::LLVM_CUSTOM_CMD {$LLVM_CUSTOM_OPT /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.vitis.ll -o $LLVM_CUSTOM_OUTPUT}

csynth_design

export_design -rtl verilog -format ip_catalog

