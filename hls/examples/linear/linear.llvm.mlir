module attributes {llvm.data_layout = "", torch.debug_module_name = "LinearOut"}  {
  llvm.mlir.global private constant @__constant_512x1000xf32(dense<0.000000e+00> : tensor<512x1000xf32>) : !llvm.array<512 x array<1000 x f32>>
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.mlir.constant(8 : index) : i64
    %17 = llvm.mlir.constant(512 : index) : i64
    %18 = llvm.mlir.constant(1000 : index) : i64
    %19 = llvm.mlir.constant(0 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %21 = llvm.mlir.constant(512 : index) : i64
    %22 = llvm.mlir.constant(1000 : index) : i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.constant(512000 : index) : i64
    %25 = llvm.mlir.null : !llvm.ptr<f32>
    %26 = llvm.getelementptr %25[%24] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %27 = llvm.ptrtoint %26 : !llvm.ptr<f32> to i64
    %28 = llvm.mlir.addressof @__constant_512x1000xf32 : !llvm.ptr<array<512 x array<1000 x f32>>>
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.getelementptr %28[%29, %29, %29] : (!llvm.ptr<array<512 x array<1000 x f32>>>, i64, i64, i64) -> !llvm.ptr<f32>
    %31 = llvm.mlir.constant(3735928559 : index) : i64
    %32 = llvm.inttoptr %31 : i64 to !llvm.ptr<f32>
    %33 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %34 = llvm.insertvalue %32, %33[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %35 = llvm.insertvalue %30, %34[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.insertvalue %36, %35[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %38 = llvm.insertvalue %21, %37[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %22, %38[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %22, %39[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %41 = llvm.insertvalue %23, %40[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%19 : i64)
  ^bb1(%42: i64):  // 2 preds: ^bb0, ^bb8
    %43 = llvm.icmp "slt" %42, %16 : i64
    llvm.cond_br %43, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%19 : i64)
  ^bb3(%44: i64):  // 2 preds: ^bb2, ^bb7
    %45 = llvm.icmp "slt" %44, %18 : i64
    llvm.cond_br %45, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%19 : i64)
  ^bb5(%46: i64):  // 2 preds: ^bb4, ^bb6
    %47 = llvm.icmp "slt" %46, %17 : i64
    llvm.cond_br %47, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %48 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.mlir.constant(512 : index) : i64
    %50 = llvm.mul %42, %49  : i64
    %51 = llvm.add %50, %46  : i64
    %52 = llvm.getelementptr %48[%51] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %53 = llvm.load %52 : !llvm.ptr<f32>
    %54 = llvm.mlir.constant(1000 : index) : i64
    %55 = llvm.mul %46, %54  : i64
    %56 = llvm.add %55, %44  : i64
    %57 = llvm.getelementptr %30[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %58 = llvm.load %57 : !llvm.ptr<f32>
    %59 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %60 = llvm.mlir.constant(1000 : index) : i64
    %61 = llvm.mul %42, %60  : i64
    %62 = llvm.add %61, %44  : i64
    %63 = llvm.getelementptr %59[%62] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %64 = llvm.load %63 : !llvm.ptr<f32>
    %65 = llvm.fmul %53, %58  : f32
    %66 = llvm.fadd %64, %65  : f32
    %67 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %68 = llvm.mlir.constant(1000 : index) : i64
    %69 = llvm.mul %42, %68  : i64
    %70 = llvm.add %69, %44  : i64
    %71 = llvm.getelementptr %67[%70] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %66, %71 : !llvm.ptr<f32>
    %72 = llvm.add %46, %20  : i64
    llvm.br ^bb5(%72 : i64)
  ^bb7:  // pred: ^bb5
    %73 = llvm.add %44, %20  : i64
    llvm.br ^bb3(%73 : i64)
  ^bb8:  // pred: ^bb3
    %74 = llvm.add %42, %20  : i64
    llvm.br ^bb1(%74 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
}
