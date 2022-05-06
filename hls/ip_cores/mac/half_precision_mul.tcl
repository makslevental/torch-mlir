set_property -dict [list \
    CONFIG.A_Precision_Type {Half} \
    CONFIG.A_TUSER_Width {1} \
    CONFIG.Add_Sub_Value {Both} \
    CONFIG.Axi_Optimize_Goal {Resources} \
    CONFIG.B_TUSER_Width {1} \
    CONFIG.C_A_Exponent_Width {5} \
    CONFIG.C_A_Fraction_Width {11} \
    CONFIG.C_Accum_Input_Msb {15} \
    CONFIG.C_Accum_Lsb {-24} \
    CONFIG.C_Accum_Msb {32} \
    CONFIG.C_BRAM_Usage {No_Usage} \
    CONFIG.C_Compare_Operation {Programmable} \
    CONFIG.C_Has_ACCUM_INPUT_OVERFLOW {false} \
    CONFIG.C_Has_ACCUM_OVERFLOW {false} \
    CONFIG.C_Has_DIVIDE_BY_ZERO {false} \
    CONFIG.C_Has_INVALID_OP {false} \
    CONFIG.C_Has_OVERFLOW {false} \
    CONFIG.C_Has_UNDERFLOW {false} \
    CONFIG.C_Latency {1} \
    CONFIG.C_Mult_Usage {Full_Usage} \
    CONFIG.C_Optimization {Speed_Optimized} \
    CONFIG.C_Rate {1} \
    CONFIG.C_Result_Exponent_Width {5} \
    CONFIG.C_Result_Fraction_Width {11} \
    CONFIG.C_TUSER_Width {1} \
    CONFIG.Component_Name {mul_half_precision} \
    CONFIG.Flow_Control {NonBlocking} \
    CONFIG.Has_ACLKEN {false} \
    CONFIG.Has_ARESETn {false} \
    CONFIG.Has_A_TLAST {false} \
    CONFIG.Has_A_TUSER {false} \
    CONFIG.Has_B_TLAST {false} \
    CONFIG.Has_B_TUSER {false} \
    CONFIG.Has_C_TLAST {false} \
    CONFIG.Has_C_TUSER {false} \
    CONFIG.Has_OPERATION_TLAST {false} \
    CONFIG.Has_OPERATION_TUSER {false} \
    CONFIG.Has_RESULT_TREADY {false} \
    CONFIG.Maximum_Latency {false} \
    CONFIG.OPERATION_TUSER_Width {1} \
    CONFIG.Operation_Type {Multiply} \
    CONFIG.RESULT_TLAST_Behv {Null} \
    CONFIG.Result_Precision_Type {Half} \
] [get_ips mul_half_precision]