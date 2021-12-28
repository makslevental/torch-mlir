module attributes {llvm.data_layout = "", torch.debug_module_name = "AdaptiveAvgPool2dOutModule"}  {
  llvm.mlir.global private constant @__constant_7x2xf32(dense<0.000000e+00> : tensor<7x2xf32>) : !llvm.array<7 x array<2 x f32>>
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %8 = llvm.insertvalue %arg5, %7[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %10 = llvm.insertvalue %arg6, %9[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %11 = llvm.insertvalue %arg10, %10[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %17 = llvm.insertvalue %arg18, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %18 = llvm.insertvalue %arg15, %17[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %19 = llvm.insertvalue %arg19, %18[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %20 = llvm.insertvalue %arg16, %19[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %21 = llvm.insertvalue %arg20, %20[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %22 = llvm.insertvalue %arg17, %21[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %23 = llvm.insertvalue %arg21, %22[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %24 = llvm.mlir.constant(400 : i64) : i64
    %25 = llvm.mlir.constant(7 : index) : i64
    %26 = llvm.mlir.constant(2 : index) : i64
    %27 = llvm.mlir.constant(20 : index) : i64
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.mlir.constant(1 : index) : i64
    %30 = llvm.mlir.constant(7 : index) : i64
    %31 = llvm.mlir.constant(2 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.mlir.constant(14 : index) : i64
    %34 = llvm.mlir.null : !llvm.ptr<f32>
    %35 = llvm.getelementptr %34[%33] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %36 = llvm.ptrtoint %35 : !llvm.ptr<f32> to i64
    %37 = llvm.mlir.addressof @__constant_7x2xf32 : !llvm.ptr<array<7 x array<2 x f32>>>
    %38 = llvm.mlir.constant(0 : index) : i64
    %39 = llvm.getelementptr %37[%38, %38, %38] : (!llvm.ptr<array<7 x array<2 x f32>>>, i64, i64, i64) -> !llvm.ptr<f32>
    %40 = llvm.mlir.constant(3735928559 : index) : i64
    %41 = llvm.inttoptr %40 : i64 to !llvm.ptr<f32>
    %42 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %44 = llvm.insertvalue %39, %43[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.insertvalue %45, %44[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %47 = llvm.insertvalue %30, %46[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %48 = llvm.insertvalue %31, %47[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %49 = llvm.insertvalue %31, %48[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %50 = llvm.insertvalue %32, %49[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    llvm.br ^bb1(%28 : i64)
  ^bb1(%51: i64):  // 2 preds: ^bb0, ^bb11
    %52 = llvm.icmp "slt" %51, %25 : i64
    llvm.cond_br %52, ^bb2, ^bb12
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%28 : i64)
  ^bb3(%53: i64):  // 2 preds: ^bb2, ^bb10
    %54 = llvm.icmp "slt" %53, %26 : i64
    llvm.cond_br %54, ^bb4, ^bb11
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%28 : i64)
  ^bb5(%55: i64):  // 2 preds: ^bb4, ^bb9
    %56 = llvm.icmp "slt" %55, %27 : i64
    llvm.cond_br %56, ^bb6, ^bb10
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%28 : i64)
  ^bb7(%57: i64):  // 2 preds: ^bb6, ^bb8
    %58 = llvm.icmp "slt" %57, %27 : i64
    llvm.cond_br %58, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %59 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %60 = llvm.mlir.constant(800 : index) : i64
    %61 = llvm.mul %51, %60  : i64
    %62 = llvm.mlir.constant(400 : index) : i64
    %63 = llvm.mul %53, %62  : i64
    %64 = llvm.add %61, %63  : i64
    %65 = llvm.mlir.constant(20 : index) : i64
    %66 = llvm.mul %55, %65  : i64
    %67 = llvm.add %64, %66  : i64
    %68 = llvm.add %67, %57  : i64
    %69 = llvm.getelementptr %59[%68] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %70 = llvm.load %69 : !llvm.ptr<f32>
    %71 = llvm.mlir.constant(2 : index) : i64
    %72 = llvm.mul %51, %71  : i64
    %73 = llvm.add %72, %53  : i64
    %74 = llvm.getelementptr %39[%73] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %75 = llvm.load %74 : !llvm.ptr<f32>
    %76 = llvm.fadd %75, %70  : f32
    %77 = llvm.mlir.constant(2 : index) : i64
    %78 = llvm.mul %51, %77  : i64
    %79 = llvm.add %78, %53  : i64
    %80 = llvm.getelementptr %39[%79] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %76, %80 : !llvm.ptr<f32>
    %81 = llvm.add %57, %29  : i64
    llvm.br ^bb7(%81 : i64)
  ^bb9:  // pred: ^bb7
    %82 = llvm.add %55, %29  : i64
    llvm.br ^bb5(%82 : i64)
  ^bb10:  // pred: ^bb5
    %83 = llvm.add %53, %29  : i64
    llvm.br ^bb3(%83 : i64)
  ^bb11:  // pred: ^bb3
    %84 = llvm.add %51, %29  : i64
    llvm.br ^bb1(%84 : i64)
  ^bb12:  // pred: ^bb1
    %85 = llvm.sitofp %24 : i64 to f32
    llvm.br ^bb13(%28 : i64)
  ^bb13(%86: i64):  // 2 preds: ^bb12, ^bb23
    %87 = llvm.icmp "slt" %86, %25 : i64
    llvm.cond_br %87, ^bb14, ^bb24
  ^bb14:  // pred: ^bb13
    llvm.br ^bb15(%28 : i64)
  ^bb15(%88: i64):  // 2 preds: ^bb14, ^bb22
    %89 = llvm.icmp "slt" %88, %26 : i64
    llvm.cond_br %89, ^bb16, ^bb23
  ^bb16:  // pred: ^bb15
    llvm.br ^bb17(%28 : i64)
  ^bb17(%90: i64):  // 2 preds: ^bb16, ^bb21
    %91 = llvm.icmp "slt" %90, %29 : i64
    llvm.cond_br %91, ^bb18, ^bb22
  ^bb18:  // pred: ^bb17
    llvm.br ^bb19(%28 : i64)
  ^bb19(%92: i64):  // 2 preds: ^bb18, ^bb20
    %93 = llvm.icmp "slt" %92, %29 : i64
    llvm.cond_br %93, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %94 = llvm.mlir.constant(2 : index) : i64
    %95 = llvm.mul %86, %94  : i64
    %96 = llvm.add %95, %88  : i64
    %97 = llvm.getelementptr %39[%96] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %98 = llvm.load %97 : !llvm.ptr<f32>
    %99 = llvm.fdiv %98, %85  : f32
    %100 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %101 = llvm.mlir.constant(2 : index) : i64
    %102 = llvm.mul %86, %101  : i64
    %103 = llvm.add %102, %88  : i64
    %104 = llvm.add %103, %90  : i64
    %105 = llvm.add %104, %92  : i64
    %106 = llvm.getelementptr %100[%105] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %99, %106 : !llvm.ptr<f32>
    %107 = llvm.add %92, %29  : i64
    llvm.br ^bb19(%107 : i64)
  ^bb21:  // pred: ^bb19
    %108 = llvm.add %90, %29  : i64
    llvm.br ^bb17(%108 : i64)
  ^bb22:  // pred: ^bb17
    %109 = llvm.add %88, %29  : i64
    llvm.br ^bb15(%109 : i64)
  ^bb23:  // pred: ^bb15
    %110 = llvm.add %86, %29  : i64
    llvm.br ^bb13(%110 : i64)
  ^bb24:  // pred: ^bb13
    llvm.return
  }
}
