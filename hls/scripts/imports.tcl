set vivado_ver [version -short]
set fpo_ver 7.1
if {[regexp -nocase {2015\.1.*} \
 $vivado_ver match]} \
 {
    set fpo_ver 7.0
} \

add_files -fileset constrs_1 -norecurse collected/alveo-u280-xdc.xdc
import_files -fileset constrs_1 collected/alveo-u280-xdc.xdc

add_files -norecurse -scan_for_includes forward.v -force
import_files -norecurse forward.v -force

add_files -norecurse -scan_for_includes collected/relu.v -force
import_files -norecurse collected/relu.v -force

add_files -norecurse -scan_for_includes collected/neg.v -force
import_files -norecurse collected/neg.v -force

add_files -norecurse -scan_for_includes collected/simple_dual_rw_ram.v -force
import_files -norecurse collected/simple_dual_rw_ram.v -force

add_files -norecurse -scan_for_includes collected/sfp_add_4_4.vhdl -force
import_files -norecurse collected/sfp_add_4_4.vhdl -force

add_files -norecurse -scan_for_includes collected/sfp_mul_4_4.vhdl -force
import_files -norecurse collected/sfp_mul_4_4.vhdl -force

read_xdc collected/clock.xdc

update_compile_order -fileset sources_1

