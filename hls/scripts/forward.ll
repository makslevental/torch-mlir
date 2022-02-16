; ModuleID = 'forward.bc'
source_filename = "/home/mlevental/server_home/dev_projects/fpga/hls/accelerator/solution1/.autopilot/db/forward.pp.0.cpp"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

@llvm.used = appending global [1 x i8*] [i8* bitcast (void ([1 x [1 x [8 x [8 x float]]]]*, [1 x [1 x [6 x [6 x float]]]]*)* @_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f to i8*)], section "llvm.metadata"

; Function Attrs: nounwind
define void @_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f([1 x [1 x [8 x [8 x float]]]]* dereferenceable(256) %arg_2, [1 x [1 x [6 x [6 x float]]]]* dereferenceable(144) %arg_3) #0 !dbg !7 {
  %arg_2.addr = alloca [1 x [1 x [8 x [8 x float]]]]*, align 8
  %arg_3.addr = alloca [1 x [1 x [6 x [6 x float]]]]*, align 8
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %sum = alloca float, align 4
  %k = alloca i32, align 4
  store [1 x [1 x [8 x [8 x float]]]]* %arg_2, [1 x [1 x [8 x [8 x float]]]]** %arg_2.addr, align 8
  call void @llvm.dbg.declare(metadata [1 x [1 x [8 x [8 x float]]]]** %arg_2.addr, metadata !21, metadata !DIExpression()), !dbg !22
  store [1 x [1 x [6 x [6 x float]]]]* %arg_3, [1 x [1 x [6 x [6 x float]]]]** %arg_3.addr, align 8
  call void @llvm.dbg.declare(metadata [1 x [1 x [6 x [6 x float]]]]** %arg_3.addr, metadata !23, metadata !DIExpression()), !dbg !24
  br label %1, !dbg !25

1:                                                ; preds = %0
  call void @llvm.dbg.declare(metadata i32* %m, metadata !26, metadata !DIExpression()), !dbg !29
  store i32 0, i32* %m, align 4, !dbg !29
  br label %2, !dbg !30

2:                                                ; preds = %32, %1
  %3 = load i32, i32* %m, align 4, !dbg !31
  %cmp = icmp slt i32 %3, 8, !dbg !33
  br i1 %cmp, label %4, label %34, !dbg !34

4:                                                ; preds = %2
  br label %5, !dbg !35

5:                                                ; preds = %4
  call void @llvm.dbg.declare(metadata i32* %n, metadata !36, metadata !DIExpression()), !dbg !39
  store i32 0, i32* %n, align 4, !dbg !39
  br label %6, !dbg !40

6:                                                ; preds = %29, %5
  %7 = load i32, i32* %n, align 4, !dbg !41
  %cmp1 = icmp slt i32 %7, 8, !dbg !43
  br i1 %cmp1, label %8, label %31, !dbg !44

8:                                                ; preds = %6
  call void @llvm.dbg.declare(metadata float* %sum, metadata !45, metadata !DIExpression()), !dbg !47
  store float 0.000000e+00, float* %sum, align 4, !dbg !47
  br label %9, !dbg !48

9:                                                ; preds = %8
  call void @llvm.dbg.declare(metadata i32* %k, metadata !49, metadata !DIExpression()), !dbg !51
  store i32 0, i32* %k, align 4, !dbg !51
  br label %10, !dbg !52

10:                                               ; preds = %22, %9
  %11 = load i32, i32* %k, align 4, !dbg !53
  %cmp2 = icmp slt i32 %11, 8, !dbg !55
  br i1 %cmp2, label %12, label %24, !dbg !56

12:                                               ; preds = %10
  %13 = load [1 x [1 x [8 x [8 x float]]]]*, [1 x [1 x [8 x [8 x float]]]]** %arg_2.addr, align 8, !dbg !57
  %arrayidx = getelementptr inbounds [1 x [1 x [8 x [8 x float]]]], [1 x [1 x [8 x [8 x float]]]]* %13, i64 0, i64 0, !dbg !57
  %arrayidx3 = getelementptr inbounds [1 x [8 x [8 x float]]], [1 x [8 x [8 x float]]]* %arrayidx, i64 0, i64 0, !dbg !57
  %14 = load i32, i32* %m, align 4, !dbg !58
  %idxprom = sext i32 %14 to i64, !dbg !57
  %arrayidx4 = getelementptr inbounds [8 x [8 x float]], [8 x [8 x float]]* %arrayidx3, i64 0, i64 %idxprom, !dbg !57
  %15 = load i32, i32* %n, align 4, !dbg !59
  %idxprom5 = sext i32 %15 to i64, !dbg !57
  %arrayidx6 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx4, i64 0, i64 %idxprom5, !dbg !57
  %16 = load float, float* %arrayidx6, align 4, !dbg !57
  %17 = load [1 x [1 x [8 x [8 x float]]]]*, [1 x [1 x [8 x [8 x float]]]]** %arg_2.addr, align 8, !dbg !60
  %arrayidx7 = getelementptr inbounds [1 x [1 x [8 x [8 x float]]]], [1 x [1 x [8 x [8 x float]]]]* %17, i64 0, i64 0, !dbg !60
  %arrayidx8 = getelementptr inbounds [1 x [8 x [8 x float]]], [1 x [8 x [8 x float]]]* %arrayidx7, i64 0, i64 0, !dbg !60
  %18 = load i32, i32* %m, align 4, !dbg !61
  %idxprom9 = sext i32 %18 to i64, !dbg !60
  %arrayidx10 = getelementptr inbounds [8 x [8 x float]], [8 x [8 x float]]* %arrayidx8, i64 0, i64 %idxprom9, !dbg !60
  %19 = load i32, i32* %n, align 4, !dbg !62
  %idxprom11 = sext i32 %19 to i64, !dbg !60
  %arrayidx12 = getelementptr inbounds [8 x float], [8 x float]* %arrayidx10, i64 0, i64 %idxprom11, !dbg !60
  %20 = load float, float* %arrayidx12, align 4, !dbg !60
  %mul = fmul float %16, %20, !dbg !63
  %21 = load float, float* %sum, align 4, !dbg !64
  %add = fadd float %21, %mul, !dbg !64
  store float %add, float* %sum, align 4, !dbg !64
  br label %22, !dbg !65

22:                                               ; preds = %12
  %23 = load i32, i32* %k, align 4, !dbg !66
  %inc = add nsw i32 %23, 1, !dbg !66
  store i32 %inc, i32* %k, align 4, !dbg !66
  br label %10, !dbg !67, !llvm.loop !68

24:                                               ; preds = %10
  %25 = load float, float* %sum, align 4, !dbg !71
  %26 = load [1 x [1 x [6 x [6 x float]]]]*, [1 x [1 x [6 x [6 x float]]]]** %arg_3.addr, align 8, !dbg !72
  %arrayidx13 = getelementptr inbounds [1 x [1 x [6 x [6 x float]]]], [1 x [1 x [6 x [6 x float]]]]* %26, i64 0, i64 0, !dbg !72
  %27 = load i32, i32* %m, align 4, !dbg !73
  %28 = load i32, i32* %n, align 4, !dbg !74
  %add14 = add nsw i32 %27, %28, !dbg !75
  %idxprom15 = sext i32 %add14 to i64, !dbg !72
  %arrayidx16 = getelementptr inbounds [1 x [6 x [6 x float]]], [1 x [6 x [6 x float]]]* %arrayidx13, i64 0, i64 %idxprom15, !dbg !72
  %arrayidx17 = getelementptr inbounds [6 x [6 x float]], [6 x [6 x float]]* %arrayidx16, i64 0, i64 0, !dbg !72
  %arrayidx18 = getelementptr inbounds [6 x float], [6 x float]* %arrayidx17, i64 0, i64 0, !dbg !72
  store float %25, float* %arrayidx18, align 4, !dbg !76
  br label %29, !dbg !77

29:                                               ; preds = %24
  %30 = load i32, i32* %n, align 4, !dbg !78
  %inc19 = add nsw i32 %30, 1, !dbg !78
  store i32 %inc19, i32* %n, align 4, !dbg !78
  br label %6, !dbg !79, !llvm.loop !80

31:                                               ; preds = %6
  br label %32, !dbg !83

32:                                               ; preds = %31
  %33 = load i32, i32* %m, align 4, !dbg !84
  %inc20 = add nsw i32 %33, 1, !dbg !84
  store i32 %inc20, i32* %m, align 4, !dbg !84
  br label %2, !dbg !85, !llvm.loop !86

34:                                               ; preds = %2
  ret void, !dbg !89
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "fpga.demangled.name"="forward" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }

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
!7 = distinct !DISubprogram(name: "forward", linkageName: "_Z7forwardRA1_A1_A8_A8_fRA1_A1_A6_A6_f", scope: !8, file: !8, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
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
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 1152, elements: !19)
!19 = !{!15, !15, !20, !20}
!20 = !DISubrange(count: 6)
!21 = !DILocalVariable(name: "arg_2", arg: 1, scope: !7, file: !8, line: 3, type: !11)
!22 = !DILocation(line: 3, column: 44, scope: !7)
!23 = !DILocalVariable(name: "arg_3", arg: 2, scope: !7, file: !8, line: 3, type: !17)
!24 = !DILocation(line: 3, column: 72, scope: !7)
!25 = !DILocation(line: 3, column: 92, scope: !7)
!26 = !DILocalVariable(name: "m", scope: !27, file: !8, line: 4, type: !28)
!27 = distinct !DILexicalBlock(scope: !7, file: !8, line: 4, column: 19)
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!29 = !DILocation(line: 4, column: 28, scope: !27)
!30 = !DILocation(line: 4, column: 24, scope: !27)
!31 = !DILocation(line: 4, column: 35, scope: !32)
!32 = distinct !DILexicalBlock(scope: !27, file: !8, line: 4, column: 19)
!33 = !DILocation(line: 4, column: 37, scope: !32)
!34 = !DILocation(line: 4, column: 19, scope: !27)
!35 = !DILocation(line: 4, column: 47, scope: !32)
!36 = !DILocalVariable(name: "n", scope: !37, file: !8, line: 5, type: !28)
!37 = distinct !DILexicalBlock(scope: !38, file: !8, line: 5, column: 21)
!38 = distinct !DILexicalBlock(scope: !32, file: !8, line: 4, column: 47)
!39 = !DILocation(line: 5, column: 30, scope: !37)
!40 = !DILocation(line: 5, column: 26, scope: !37)
!41 = !DILocation(line: 5, column: 37, scope: !42)
!42 = distinct !DILexicalBlock(scope: !37, file: !8, line: 5, column: 21)
!43 = !DILocation(line: 5, column: 39, scope: !42)
!44 = !DILocation(line: 5, column: 21, scope: !37)
!45 = !DILocalVariable(name: "sum", scope: !46, file: !8, line: 6, type: !13)
!46 = distinct !DILexicalBlock(scope: !42, file: !8, line: 5, column: 49)
!47 = !DILocation(line: 6, column: 13, scope: !46)
!48 = !DILocation(line: 6, column: 7, scope: !46)
!49 = !DILocalVariable(name: "k", scope: !50, file: !8, line: 7, type: !28)
!50 = distinct !DILexicalBlock(scope: !46, file: !8, line: 7, column: 23)
!51 = !DILocation(line: 7, column: 32, scope: !50)
!52 = !DILocation(line: 7, column: 28, scope: !50)
!53 = !DILocation(line: 7, column: 39, scope: !54)
!54 = distinct !DILexicalBlock(scope: !50, file: !8, line: 7, column: 23)
!55 = !DILocation(line: 7, column: 41, scope: !54)
!56 = !DILocation(line: 7, column: 23, scope: !50)
!57 = !DILocation(line: 8, column: 16, scope: !54)
!58 = !DILocation(line: 8, column: 28, scope: !54)
!59 = !DILocation(line: 8, column: 31, scope: !54)
!60 = !DILocation(line: 8, column: 36, scope: !54)
!61 = !DILocation(line: 8, column: 48, scope: !54)
!62 = !DILocation(line: 8, column: 51, scope: !54)
!63 = !DILocation(line: 8, column: 34, scope: !54)
!64 = !DILocation(line: 8, column: 13, scope: !54)
!65 = !DILocation(line: 8, column: 9, scope: !54)
!66 = !DILocation(line: 7, column: 46, scope: !54)
!67 = !DILocation(line: 7, column: 23, scope: !54)
!68 = distinct !{!68, !56, !69, !70}
!69 = !DILocation(line: 8, column: 52, scope: !50)
!70 = !{!"llvm.loop.name", !"VITIS_LOOP_7_3"}
!71 = !DILocation(line: 9, column: 29, scope: !46)
!72 = !DILocation(line: 9, column: 7, scope: !46)
!73 = !DILocation(line: 9, column: 16, scope: !46)
!74 = !DILocation(line: 9, column: 18, scope: !46)
!75 = !DILocation(line: 9, column: 17, scope: !46)
!76 = !DILocation(line: 9, column: 27, scope: !46)
!77 = !DILocation(line: 10, column: 5, scope: !46)
!78 = !DILocation(line: 5, column: 44, scope: !42)
!79 = !DILocation(line: 5, column: 21, scope: !42)
!80 = distinct !{!80, !44, !81, !82}
!81 = !DILocation(line: 10, column: 5, scope: !37)
!82 = !{!"llvm.loop.name", !"VITIS_LOOP_5_2"}
!83 = !DILocation(line: 11, column: 3, scope: !38)
!84 = !DILocation(line: 4, column: 42, scope: !32)
!85 = !DILocation(line: 4, column: 19, scope: !32)
!86 = distinct !{!86, !34, !87, !88}
!87 = !DILocation(line: 11, column: 3, scope: !27)
!88 = !{!"llvm.loop.name", !"VITIS_LOOP_4_1"}
!89 = !DILocation(line: 12, column: 3, scope: !7)
