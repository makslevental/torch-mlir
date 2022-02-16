; ModuleID = '/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.ll'
source_filename = "LLVMDialectModule"

@__constant_1xf32 = private constant [1 x float] [float 1.000000e+00]
@__constant_1x1x3x3xf32 = private constant [1 x [1 x [3 x [3 x float]]]] [[1 x [3 x [3 x float]]] [[3 x [3 x float]] [[3 x float] [float 1.000000e+00, float 1.000000e+00, float 1.000000e+00], [3 x float] [float 1.000000e+00, float 1.000000e+00, float 1.000000e+00], [3 x float] [float 1.000000e+00, float 1.000000e+00, float 1.000000e+00]]]]

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @abort()

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ([1 x [1 x [8 x [8 x float]]]]*, [1 x [1 x [6 x [6 x float]]]]*)* @_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f to i8*)], section "llvm.metadata"

; Function Attrs: nounwind
define void @_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f([1 x [1 x [8 x [8 x float]]]]* dereferenceable(256) %arg_2, [1 x [1 x [6 x [6 x float]]]]* dereferenceable(144) %arg_3) #0 {
bb_0:
  br i1 true, label %bb_1, label %bb_35

bb_1:                                             ; preds = %bb_0
  br label %bb_2

bb_2:                                             ; preds = %bb_12, %bb_1
  %val_1 = phi i64 [ %val_22, %bb_12 ], [ 0, %bb_1 ]
  %val_2 = icmp slt i64 %val_1, 1
  br i1 %val_2, label %bb_3, label %bb_13

bb_3:                                             ; preds = %bb_2
  br label %bb_4

bb_4:                                             ; preds = %bb_11, %bb_3
  %val_3 = phi i64 [ %val_21, %bb_11 ], [ 0, %bb_3 ]
  %val_4 = icmp slt i64 %val_3, 1
  br i1 %val_4, label %bb_5, label %bb_12

bb_5:                                             ; preds = %bb_4
  br label %bb_6

bb_6:                                             ; preds = %bb_10, %bb_5
  %val_5 = phi i64 [ %val_20, %bb_10 ], [ 0, %bb_5 ]
  %val_6 = icmp slt i64 %val_5, 6
  br i1 %val_6, label %bb_7, label %bb_11

bb_7:                                             ; preds = %bb_6
  br label %bb_8

bb_8:                                             ; preds = %bb_9, %bb_7
  %val_7 = phi i64 [ %val_19, %bb_9 ], [ 0, %bb_7 ]
  %val_8 = icmp slt i64 %val_7, 6
  br i1 %val_8, label %bb_9, label %bb_10

bb_9:                                             ; preds = %bb_8
  %val_9 = add i64 %val_1, %val_3
  %val_10 = mul i64 %val_9, 36
  %val_11 = mul i64 %val_5, 6
  %val_12 = add i64 %val_10, %val_11
  %val_13 = add i64 %val_12, %val_7
  %val_14 = udiv i64 %val_13, 36
  %val_15 = urem i64 %val_13, 36
  %val_16 = udiv i64 %val_15, 6
  %val_17 = urem i64 %val_15, 6
  %val_18 = getelementptr inbounds [1 x [1 x [6 x [6 x float]]]], [1 x [1 x [6 x [6 x float]]]]* %arg_3, i64 0, i64 %val_14, i64 0, i64 %val_16, i64 %val_17
  store float 1.000000e+00, float* %val_18, align 4
  %val_19 = add i64 %val_7, 1
  br label %bb_8

bb_10:                                            ; preds = %bb_8
  %val_20 = add i64 %val_5, 1
  br label %bb_6

bb_11:                                            ; preds = %bb_6
  %val_21 = add i64 %val_3, 1
  br label %bb_4

bb_12:                                            ; preds = %bb_4
  %val_22 = add i64 %val_1, 1
  br label %bb_2

bb_13:                                            ; preds = %bb_2
  br label %bb_14

bb_14:                                            ; preds = %bb_33, %bb_13
  %val_23 = phi i64 [ %val_86, %bb_33 ], [ 0, %bb_13 ]
  %val_24 = icmp slt i64 %val_23, 1
  br i1 %val_24, label %bb_15, label %bb_34

bb_15:                                            ; preds = %bb_14
  br label %bb_16

bb_16:                                            ; preds = %bb_32, %bb_15
  %val_25 = phi i64 [ %val_85, %bb_32 ], [ 0, %bb_15 ]
  %val_26 = icmp slt i64 %val_25, 1
  br i1 %val_26, label %bb_17, label %bb_33

bb_17:                                            ; preds = %bb_16
  br label %bb_18

bb_18:                                            ; preds = %bb_31, %bb_17
  %val_27 = phi i64 [ %val_84, %bb_31 ], [ 0, %bb_17 ]
  %val_28 = icmp slt i64 %val_27, 6
  br i1 %val_28, label %bb_19, label %bb_32

bb_19:                                            ; preds = %bb_18
  br label %bb_20

bb_20:                                            ; preds = %bb_30, %bb_19
  %val_29 = phi i64 [ %val_83, %bb_30 ], [ 0, %bb_19 ]
  %val_30 = icmp slt i64 %val_29, 6
  br i1 %val_30, label %bb_21, label %bb_31

bb_21:                                            ; preds = %bb_20
  br label %bb_22

bb_22:                                            ; preds = %bb_29, %bb_21
  %val_31 = phi i64 [ %val_82, %bb_29 ], [ 0, %bb_21 ]
  %val_32 = icmp slt i64 %val_31, 1
  br i1 %val_32, label %bb_23, label %bb_30

bb_23:                                            ; preds = %bb_22
  br label %bb_24

bb_24:                                            ; preds = %bb_28, %bb_23
  %val_33 = phi i64 [ %val_81, %bb_28 ], [ 0, %bb_23 ]
  %val_34 = icmp slt i64 %val_33, 3
  br i1 %val_34, label %bb_25, label %bb_29

bb_25:                                            ; preds = %bb_24
  br label %bb_26

bb_26:                                            ; preds = %bb_27, %bb_25
  %val_35 = phi i64 [ %val_80, %bb_27 ], [ 0, %bb_25 ]
  %val_36 = icmp slt i64 %val_35, 3
  br i1 %val_36, label %bb_27, label %bb_28

bb_27:                                            ; preds = %bb_26
  %val_37 = add i64 %val_27, %val_33
  %val_38 = add i64 %val_29, %val_35
  %val_39 = add i64 %val_23, %val_31
  %val_40 = shl i64 %val_39, 6
  %val_41 = shl i64 %val_37, 3
  %val_42 = add i64 %val_40, %val_41
  %val_43 = add i64 %val_42, %val_38
  %val_44 = lshr i64 %val_43, 6
  %val_45 = lshr i64 %val_43, 3
  %val_46 = and i64 %val_45, 7
  %val_47 = and i64 %val_43, 7
  %val_48 = getelementptr inbounds [1 x [1 x [8 x [8 x float]]]], [1 x [1 x [8 x [8 x float]]]]* %arg_2, i64 0, i64 %val_44, i64 0, i64 %val_46, i64 %val_47
  %val_49 = load float, float* %val_48, align 4
  %val_50 = add i64 %val_25, %val_31
  %val_51 = mul i64 %val_50, 9
  %val_52 = mul i64 %val_33, 3
  %val_53 = add i64 %val_51, %val_52
  %val_54 = add i64 %val_53, %val_35
  %val_55 = getelementptr [1 x [1 x [3 x [3 x float]]]], [1 x [1 x [3 x [3 x float]]]]* @__constant_1x1x3x3xf32, i64 0, i64 0, i64 0, i64 0, i64 %val_54
  %val_56 = load float, float* %val_55, align 4
  %val_57 = add i64 %val_23, %val_25
  %val_58 = mul i64 %val_57, 36
  %val_59 = mul i64 %val_27, 6
  %val_60 = add i64 %val_58, %val_59
  %val_61 = add i64 %val_60, %val_29
  %val_62 = udiv i64 %val_61, 36
  %val_63 = urem i64 %val_61, 36
  %val_64 = udiv i64 %val_63, 6
  %val_65 = urem i64 %val_63, 6
  %val_66 = getelementptr inbounds [1 x [1 x [6 x [6 x float]]]], [1 x [1 x [6 x [6 x float]]]]* %arg_3, i64 0, i64 %val_62, i64 0, i64 %val_64, i64 %val_65
  %val_67 = load float, float* %val_66, align 4
  %val_68 = fmul float %val_49, %val_56
  %val_69 = fadd float %val_67, %val_68
  %val_70 = add i64 %val_23, %val_25
  %val_71 = mul i64 %val_70, 36
  %val_72 = mul i64 %val_27, 6
  %val_73 = add i64 %val_71, %val_72
  %val_74 = add i64 %val_73, %val_29
  %val_75 = udiv i64 %val_74, 36
  %val_76 = urem i64 %val_74, 36
  %val_77 = udiv i64 %val_76, 6
  %val_78 = urem i64 %val_76, 6
  %val_79 = getelementptr inbounds [1 x [1 x [6 x [6 x float]]]], [1 x [1 x [6 x [6 x float]]]]* %arg_3, i64 0, i64 %val_75, i64 0, i64 %val_77, i64 %val_78
  store float %val_69, float* %val_79, align 4
  %val_80 = add i64 %val_35, 1
  br label %bb_26

bb_28:                                            ; preds = %bb_26
  %val_81 = add i64 %val_33, 1
  br label %bb_24

bb_29:                                            ; preds = %bb_24
  %val_82 = add i64 %val_31, 1
  br label %bb_22

bb_30:                                            ; preds = %bb_22
  %val_83 = add i64 %val_29, 1
  br label %bb_20

bb_31:                                            ; preds = %bb_20
  %val_84 = add i64 %val_27, 1
  br label %bb_18

bb_32:                                            ; preds = %bb_18
  %val_85 = add i64 %val_25, 1
  br label %bb_16

bb_33:                                            ; preds = %bb_16
  %val_86 = add i64 %val_23, 1
  br label %bb_14

bb_34:                                            ; preds = %bb_14
  ret void

bb_35:                                            ; preds = %bb_0
  unreachable
}


attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "fpga.demangled.name"="forward" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/home/mlevental/server_home/dev_projects/fpga/hls/accelerator/solution1/.autopilot/db/forward.pp.0.cpp", directory: "/home/mlevental/server_home/dev_projects/fpga/hls")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 "}
!7 = distinct !DISubprogram(name: "forward", linkageName: "_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f", scope: !8, file: !8, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, unit: !0)
!8 = !DIFile(filename: "forward.cpp", directory: "/home/mlevental/server_home/dev_projects/fpga/hls")
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !17}
!11 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !12, size: 64)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 2048, elements: !14)
!13 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!14 = !{!15, !15, !16, !16}
!15 = !DISubrange(count: 1)
!16 = !DISubrange(count: 8)
!17 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !18, size: 64)
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 73728, elements: !19)
!19 = !{!15, !20, !21, !21}
!20 = !DISubrange(count: 64)
!21 = !DISubrange(count: 6)
