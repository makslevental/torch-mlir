module attributes {llvm.data_layout = "", torch.debug_module_name = "MatmulDotOut"}  {
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(5 : index) : i64
    %2 = llvm.mlir.constant(10 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%3 : i64)
  ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb8
    %6 = llvm.icmp "slt" %5, %0 : i64
    llvm.cond_br %6, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%3 : i64)
  ^bb3(%7: i64):  // 2 preds: ^bb2, ^bb7
    %8 = llvm.icmp "slt" %7, %2 : i64
    llvm.cond_br %8, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%3 : i64)
  ^bb5(%9: i64):  // 2 preds: ^bb4, ^bb6
    %10 = llvm.icmp "slt" %9, %1 : i64
    llvm.cond_br %10, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %11 = llvm.mlir.constant(5 : index) : i64
    %12 = llvm.mul %5, %11  : i64
    %13 = llvm.add %12, %9  : i64
    %14 = llvm.getelementptr %arg1[%13] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %15 = llvm.load %14 : !llvm.ptr<f32>
    %16 = llvm.mlir.constant(10 : index) : i64
    %17 = llvm.mul %9, %16  : i64
    %18 = llvm.add %17, %7  : i64
    %19 = llvm.getelementptr %arg8[%18] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %20 = llvm.load %19 : !llvm.ptr<f32>
    %21 = llvm.mlir.constant(10 : index) : i64
    %22 = llvm.mul %5, %21  : i64
    %23 = llvm.add %22, %7  : i64
    %24 = llvm.getelementptr %arg15[%23] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %25 = llvm.load %24 : !llvm.ptr<f32>
    %26 = llvm.fmul %15, %20  : f32
    %27 = llvm.fadd %25, %26  : f32
    %28 = llvm.mlir.constant(10 : index) : i64
    %29 = llvm.mul %5, %28  : i64
    %30 = llvm.add %29, %7  : i64
    %31 = llvm.getelementptr %arg15[%30] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %27, %31 : !llvm.ptr<f32>
    %32 = llvm.add %9, %4  : i64
    llvm.br ^bb5(%32 : i64)
  ^bb7:  // pred: ^bb5
    %33 = llvm.add %7, %4  : i64
    llvm.br ^bb3(%33 : i64)
  ^bb8:  // pred: ^bb3
    %34 = llvm.add %5, %4  : i64
    llvm.br ^bb1(%34 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}

