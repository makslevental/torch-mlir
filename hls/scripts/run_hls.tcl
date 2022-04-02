open_project -reset proj
add_files wrapper.cpp
set_top forward

open_solution -reset solution1
#set_part "xcvu35p-fsvh2104-3-e"
#set_part "xc7a200tfbg484-3"
#set_part "xcvu19p-fsvb3824-3-e"
set_part "xcvu57p_CIV-fsvk2892-3-e"
create_clock -period "680MHz"


set ::LLVM_CUSTOM_CMD {$LLVM_CUSTOM_OPT XXX_DIR_XXX/XXX_LL_FILE_XXX -o $LLVM_CUSTOM_OUTPUT}

proc print {args} {
    set cmd [lindex $args 0]
    puts stdout "$> $cmd"
}

#trace add execution source enterstep print
#trace add execution csynth_design enterstep print
#trace add execution transform enterstep print


csynth_design

#export_design -rtl verilog -format ip_catalog

