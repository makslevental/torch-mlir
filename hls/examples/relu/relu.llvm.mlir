module attributes {llvm.data_layout = "", torch.debug_module_name = "ReLUModule"}  {
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %9 = llvm.mlir.constant(10 : index) : i64
    %10 = llvm.mlir.constant(0 : index) : i64
    %11 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%10 : i64)
  ^bb1(%12: i64):  // 2 preds: ^bb0, ^bb5
    %13 = llvm.icmp "slt" %12, %9 : i64
    llvm.cond_br %13, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%10 : i64)
  ^bb3(%14: i64):  // 2 preds: ^bb2, ^bb4
    %15 = llvm.icmp "slt" %14, %9 : i64
    llvm.cond_br %15, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %16 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.mlir.constant(10 : index) : i64
    %18 = llvm.mul %12, %17  : i64
    %19 = llvm.add %18, %14  : i64
    %20 = llvm.getelementptr %16[%19] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %21 = llvm.load %20 : !llvm.ptr<f32>
    %22 = llvm.fcmp "ugt" %21, %8 : f32
    %23 = llvm.select %22, %21, %8 : i1, f32
    %24 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %25 = llvm.mlir.constant(10 : index) : i64
    %26 = llvm.mul %12, %25  : i64
    %27 = llvm.add %26, %14  : i64
    %28 = llvm.getelementptr %24[%27] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %23, %28 : !llvm.ptr<f32>
    %29 = llvm.add %14, %11  : i64
    llvm.br ^bb3(%29 : i64)
  ^bb5:  // pred: ^bb3
    %30 = llvm.add %12, %11  : i64
    llvm.br ^bb1(%30 : i64)
  ^bb6:  // pred: ^bb1
    llvm.return
  }
}
