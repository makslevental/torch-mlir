source_filename = "LLVMDialectModule"
declare float @expf(float)
declare float @llvm.fmuladd.f32(float %a, float %b, float %c)
define void @forward(float %_arg0_0_0_0_0, float %_arg0_0_0_0_1, float %_arg0_0_0_0_2, float %_arg0_0_0_0_3, float %_arg0_0_0_0_4, float %_arg0_0_0_0_5, float %_arg0_0_0_0_6, float %_arg0_0_0_0_7, float %_arg0_0_0_0_8, float %_arg0_0_0_0_9, float %_arg0_0_0_0_10, float %_arg0_0_0_1_0, float %_arg0_0_0_1_1, float %_arg0_0_0_1_2, float %_arg0_0_0_1_3, float %_arg0_0_0_1_4, float %_arg0_0_0_1_5, float %_arg0_0_0_1_6, float %_arg0_0_0_1_7, float %_arg0_0_0_1_8, float %_arg0_0_0_1_9, float %_arg0_0_0_1_10, float %_arg0_0_0_2_0, float %_arg0_0_0_2_1, float %_arg0_0_0_2_2, float %_arg0_0_0_2_3, float %_arg0_0_0_2_4, float %_arg0_0_0_2_5, float %_arg0_0_0_2_6, float %_arg0_0_0_2_7, float %_arg0_0_0_2_8, float %_arg0_0_0_2_9, float %_arg0_0_0_2_10, float %_arg0_0_0_3_0, float %_arg0_0_0_3_1, float %_arg0_0_0_3_2, float %_arg0_0_0_3_3, float* %_arg1_0_0_0_0, float* %_arg1_0_0_0_1, float* %_arg1_0_0_0_2, float* %_arg1_0_0_0_3, float* %_arg1_0_0_0_4, float* %_arg1_0_0_0_5, float* %_arg1_0_0_0_6, float* %_arg1_0_0_0_7, float* %_arg1_0_0_0_8, float* %_arg1_0_0_1_0, float* %_arg1_0_0_1_1) {

%val_3 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_0, float 1107.0, float 53081.0)
%val_6 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_1, float 35658.0, float %val_3)
%val_9 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_2, float 89494.0, float %val_6)
%val_12 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_0, float 18625.0, float %val_9)
%val_15 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_1, float 46195.0, float %val_12)
%val_18 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_2, float 27698.0, float %val_15)
%val_21 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_0, float 24441.0, float %val_18)
%val_24 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_1, float 94326.0, float %val_21)
%val_27 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_2, float 39925.0, float %val_24)
%val_28 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_1, float 1107.0, float 53081.0)
%val_29 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_2, float 35658.0, float %val_28)
%val_31 = call float @llvm.fmuladd.f32(float %_arg0_0_0_0_3, float 89494.0, float %val_29)
%val_32 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_1, float 18625.0, float %val_31)
%val_33 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_2, float 46195.0, float %val_32)
%val_35 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_3, float 27698.0, float %val_33)
%val_36 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_1, float 24441.0, float %val_35)
%val_37 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_2, float 94326.0, float %val_36)
%val_39 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_3, float 39925.0, float %val_37)
%val_40 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_0, float 1107.0, float 53081.0)
%val_41 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_1, float 35658.0, float %val_40)
%val_42 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_2, float 89494.0, float %val_41)
%val_43 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_0, float 18625.0, float %val_42)
%val_44 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_1, float 46195.0, float %val_43)
%val_45 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_2, float 27698.0, float %val_44)
%val_47 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_0, float 24441.0, float %val_45)
%val_49 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_1, float 94326.0, float %val_47)
%val_51 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_2, float 39925.0, float %val_49)
%val_52 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_1, float 1107.0, float 53081.0)
%val_53 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_2, float 35658.0, float %val_52)
%val_54 = call float @llvm.fmuladd.f32(float %_arg0_0_0_1_3, float 89494.0, float %val_53)
%val_55 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_1, float 18625.0, float %val_54)
%val_56 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_2, float 46195.0, float %val_55)
%val_57 = call float @llvm.fmuladd.f32(float %_arg0_0_0_2_3, float 27698.0, float %val_56)
%val_58 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_1, float 24441.0, float %val_57)
%val_59 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_2, float 94326.0, float %val_58)
%val_61 = call float @llvm.fmuladd.f32(float %_arg0_0_0_3_3, float 39925.0, float %val_59)
store float %val_27, float* %_arg1_0_0_0_0, align 4
store float %val_39, float* %_arg1_0_0_0_1, align 4
store float %val_51, float* %_arg1_0_0_1_0, align 4
store float %val_61, float* %_arg1_0_0_1_1, align 4
ret void
}
