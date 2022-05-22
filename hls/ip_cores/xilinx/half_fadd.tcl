create_ip -name floating_point -version $fpo_ver -vendor xilinx.com -library ip -module_name half_fadd

set_property -dict [list\
    CONFIG.A_Precision_Type {Half}\
    CONFIG.Flow_Control {NonBlocking}\
    CONFIG.Maximum_Latency {false}\
    CONFIG.C_Latency {1}\
    CONFIG.Result_Precision_Type {Half}\
    CONFIG.C_Result_Exponent_Width {5}\
    CONFIG.C_Result_Fraction_Width {11}\
    CONFIG.C_Accum_Msb {32}\
    CONFIG.C_Accum_Lsb {-24}\
    CONFIG.C_Accum_Input_Msb {15}\
    CONFIG.C_Mult_Usage {Full_Usage}\
    CONFIG.Has_RESULT_TREADY {false}\
    CONFIG.C_Rate {1}\
    CONFIG.Add_Sub_Value {Both}\
    CONFIG.component_name half_fadd\
]  -objects [get_ips half_fadd]

add_files -norecurse -scan_for_includes half_fadd.v -force
import_files -norecurse half_fadd.v -force
