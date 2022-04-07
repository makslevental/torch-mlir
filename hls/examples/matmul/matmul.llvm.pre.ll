; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @forward(float* %0, float* %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, float* %7, float* %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, float* %14, float* %15, i64 %16, i64 %17, i64 %18, i64 %19, i64 %20) !dbg !3 {
br label %22, !dbg !7

22:                                               ; preds = %54, %21
%23 = phi i64 [ %55, %54 ], [ 0, %21 ]
%24 = icmp slt i64 %23, 4, !dbg !9
br i1 %24, label %25, label %56, !dbg !10

25:                                               ; preds = %22
br label %26, !dbg !11

26:                                               ; preds = %52, %25
%27 = phi i64 [ %53, %52 ], [ 0, %25 ]
%28 = icmp slt i64 %27, 10, !dbg !12
br i1 %28, label %29, label %54, !dbg !13

29:                                               ; preds = %26
br label %30, !dbg !14

30:                                               ; preds = %33, %29
%31 = phi i64 [ %51, %33 ], [ 0, %29 ]
%32 = icmp slt i64 %31, 5, !dbg !15
br i1 %32, label %33, label %52, !dbg !16

33:                                               ; preds = %30
%34 = mul i64 %23, 5, !dbg !17
%35 = add i64 %34, %31, !dbg !18
%36 = getelementptr float, float* %1, i64 %35, !dbg !19
%37 = load float, float* %36, align 4, !dbg !20
%38 = mul i64 %31, 10, !dbg !21
%39 = add i64 %38, %27, !dbg !22
%40 = getelementptr float, float* %8, i64 %39, !dbg !23
%41 = load float, float* %40, align 4, !dbg !24
%42 = mul i64 %23, 10, !dbg !25
%43 = add i64 %42, %27, !dbg !26
%44 = getelementptr float, float* %15, i64 %43, !dbg !27
%45 = load float, float* %44, align 4, !dbg !28
%46 = fmul float %37, %41, !dbg !29
%47 = fadd float %45, %46, !dbg !30
%48 = mul i64 %23, 10, !dbg !31
%49 = add i64 %48, %27, !dbg !32
%50 = getelementptr float, float* %15, i64 %49, !dbg !33
store float %47, float* %50, align 4, !dbg !34
%51 = add i64 %31, 1, !dbg !35
br label %30, !dbg !36

52:                                               ; preds = %30
%53 = add i64 %27, 1, !dbg !37
br label %26, !dbg !38

54:                                               ; preds = %26
%55 = add i64 %23, 1, !dbg !39
br label %22, !dbg !40

56:                                               ; preds = %22
ret void, !dbg !41
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "forward", linkageName: "forward", scope: null, file: !4, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "vitis_stuff/matmul.llvm.pre.mlir.opt.ll", directory: "/home/mlevental/dev_projects/torch-mlir/hls")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 8, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 10, column: 10, scope: !8)
!10 = !DILocation(line: 11, column: 5, scope: !8)
!11 = !DILocation(line: 13, column: 5, scope: !8)
!12 = !DILocation(line: 15, column: 10, scope: !8)
!13 = !DILocation(line: 16, column: 5, scope: !8)
!14 = !DILocation(line: 18, column: 5, scope: !8)
!15 = !DILocation(line: 20, column: 11, scope: !8)
!16 = !DILocation(line: 21, column: 5, scope: !8)
!17 = !DILocation(line: 24, column: 11, scope: !8)
!18 = !DILocation(line: 25, column: 11, scope: !8)
!19 = !DILocation(line: 26, column: 11, scope: !8)
!20 = !DILocation(line: 27, column: 11, scope: !8)
!21 = !DILocation(line: 29, column: 11, scope: !8)
!22 = !DILocation(line: 30, column: 11, scope: !8)
!23 = !DILocation(line: 31, column: 11, scope: !8)
!24 = !DILocation(line: 32, column: 11, scope: !8)
!25 = !DILocation(line: 34, column: 11, scope: !8)
!26 = !DILocation(line: 35, column: 11, scope: !8)
!27 = !DILocation(line: 36, column: 11, scope: !8)
!28 = !DILocation(line: 37, column: 11, scope: !8)
!29 = !DILocation(line: 38, column: 11, scope: !8)
!30 = !DILocation(line: 39, column: 11, scope: !8)
!31 = !DILocation(line: 41, column: 11, scope: !8)
!32 = !DILocation(line: 42, column: 11, scope: !8)
!33 = !DILocation(line: 43, column: 11, scope: !8)
!34 = !DILocation(line: 44, column: 5, scope: !8)
!35 = !DILocation(line: 45, column: 11, scope: !8)
!36 = !DILocation(line: 46, column: 5, scope: !8)
!37 = !DILocation(line: 48, column: 11, scope: !8)
!38 = !DILocation(line: 49, column: 5, scope: !8)
!39 = !DILocation(line: 51, column: 11, scope: !8)
!40 = !DILocation(line: 52, column: 5, scope: !8)
!41 = !DILocation(line: 54, column: 5, scope: !8)
