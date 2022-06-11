set vivado_ver [version -short]
set fpo_ver 7.1
if {[regexp -nocase {2015\.1.*} \
 $vivado_ver match]} \
 {
    set fpo_ver 7.0
} \

add_files -fileset constrs_1 -norecurse alveo-u280-xdc.xdc
import_files -fileset constrs_1 alveo-u280-xdc.xdc

add_files -norecurse -scan_for_includes forward.v -force
import_files -norecurse forward.v -force

add_files -norecurse -scan_for_includes relu.v -force
import_files -norecurse relu.v -force

add_files -norecurse -scan_for_includes neg.v -force
import_files -norecurse neg.v -force

add_files -norecurse -scan_for_includes simple_dual_rw_ram.v -force
import_files -norecurse simple_dual_rw_ram.v -force

source half_fadd.tcl
source half_fmul.tcl

#add_files -norecurse -scan_for_includes top_add_11ns_11ns_11_2_1.v -force
#import_files -norecurse top_add_11ns_11ns_11_2_1.v -force
#add_files -norecurse -scan_for_includes top_mul_mul_11s_11s_16_4_1.v -force
#import_files -norecurse top_mul_mul_11s_11s_16_4_1.v -force

read_xdc clock.xdc

update_compile_order -fileset sources_1

