module attributes {llvm.data_layout = "", torch.debug_module_name = "BatchNorm2dOut"}  {
  llvm.func @abort()
  llvm.mlir.global private constant @__constant_4xf32_0(dense<1.000000e+00> : tensor<4xf32>) : !llvm.array<4 x f32>
  llvm.mlir.global private constant @__constant_4xf32(dense<0.000000e+00> : tensor<4xf32>) : !llvm.array<4 x f32>
  llvm.func @forward(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr<f32>, %arg23: !llvm.ptr<f32>, %arg24: i64, %arg25: i64, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: !llvm.ptr<f32>, %arg30: !llvm.ptr<f32>, %arg31: i64, %arg32: i64, %arg33: i64, %arg34: i64, %arg35: i64) {
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
    %12 = llvm.mlir.constant(1.000000e-05 : f64) : f64
    %13 = llvm.mlir.constant(true) : i1
    %14 = llvm.mlir.constant(1 : index) : i64
    %15 = llvm.mlir.constant(4 : index) : i64
    %16 = llvm.mlir.constant(10 : index) : i64
    %17 = llvm.mlir.constant(0 : index) : i64
    %18 = llvm.mlir.constant(4 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.null : !llvm.ptr<f32>
    %21 = llvm.getelementptr %20[%18] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %22 = llvm.ptrtoint %21 : !llvm.ptr<f32> to i64
    %23 = llvm.mlir.addressof @__constant_4xf32 : !llvm.ptr<array<4 x f32>>
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.getelementptr %23[%24, %24] : (!llvm.ptr<array<4 x f32>>, i64, i64) -> !llvm.ptr<f32>
    %26 = llvm.mlir.constant(3735928559 : index) : i64
    %27 = llvm.inttoptr %26 : i64 to !llvm.ptr<f32>
    %28 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %29 = llvm.insertvalue %27, %28[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %30 = llvm.insertvalue %25, %29[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.mlir.constant(0 : index) : i64
    %32 = llvm.insertvalue %31, %30[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %33 = llvm.insertvalue %18, %32[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %34 = llvm.insertvalue %19, %33[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %35 = llvm.mlir.constant(4 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.null : !llvm.ptr<f32>
    %38 = llvm.getelementptr %37[%35] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %39 = llvm.ptrtoint %38 : !llvm.ptr<f32> to i64
    %40 = llvm.mlir.addressof @__constant_4xf32_0 : !llvm.ptr<array<4 x f32>>
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.getelementptr %40[%41, %41] : (!llvm.ptr<array<4 x f32>>, i64, i64) -> !llvm.ptr<f32>
    %43 = llvm.mlir.constant(3735928559 : index) : i64
    %44 = llvm.inttoptr %43 : i64 to !llvm.ptr<f32>
    %45 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %46 = llvm.insertvalue %44, %45[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %47 = llvm.insertvalue %42, %46[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.insertvalue %48, %47[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %50 = llvm.insertvalue %35, %49[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %51 = llvm.insertvalue %36, %50[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    llvm.cond_br %13, ^bb1, ^bb14
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2(%17 : i64)
  ^bb2(%52: i64):  // 2 preds: ^bb1, ^bb12
    %53 = llvm.icmp "slt" %52, %14 : i64
    llvm.cond_br %53, ^bb3, ^bb13
  ^bb3:  // pred: ^bb2
    llvm.br ^bb4(%17 : i64)
  ^bb4(%54: i64):  // 2 preds: ^bb3, ^bb11
    %55 = llvm.icmp "slt" %54, %15 : i64
    llvm.cond_br %55, ^bb5, ^bb12
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6(%17 : i64)
  ^bb6(%56: i64):  // 2 preds: ^bb5, ^bb10
    %57 = llvm.icmp "slt" %56, %16 : i64
    llvm.cond_br %57, ^bb7, ^bb11
  ^bb7:  // pred: ^bb6
    llvm.br ^bb8(%17 : i64)
  ^bb8(%58: i64):  // 2 preds: ^bb7, ^bb9
    %59 = llvm.icmp "slt" %58, %16 : i64
    llvm.cond_br %59, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    %60 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %61 = llvm.mlir.constant(400 : index) : i64
    %62 = llvm.mul %52, %61  : i64
    %63 = llvm.mlir.constant(100 : index) : i64
    %64 = llvm.mul %54, %63  : i64
    %65 = llvm.add %62, %64  : i64
    %66 = llvm.mlir.constant(10 : index) : i64
    %67 = llvm.mul %56, %66  : i64
    %68 = llvm.add %65, %67  : i64
    %69 = llvm.add %68, %58  : i64
    %70 = llvm.getelementptr %60[%69] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %71 = llvm.load %70 : !llvm.ptr<f32>
    %72 = llvm.getelementptr %42[%54] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %73 = llvm.load %72 : !llvm.ptr<f32>
    %74 = llvm.getelementptr %25[%54] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %75 = llvm.load %74 : !llvm.ptr<f32>
    %76 = llvm.fsub %71, %75  : f32
    %77 = llvm.fptrunc %12 : f64 to f32
    %78 = llvm.fadd %73, %77  : f32
    %79 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %80 = "llvm.intr.sqrt"(%78) : (f32) -> f32
    %81 = llvm.fdiv %79, %80  : f32
    %82 = llvm.fmul %76, %81  : f32
    %83 = llvm.fmul %82, %73  : f32
    %84 = llvm.fadd %83, %75  : f32
    %85 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %86 = llvm.mlir.constant(400 : index) : i64
    %87 = llvm.mul %52, %86  : i64
    %88 = llvm.mlir.constant(100 : index) : i64
    %89 = llvm.mul %54, %88  : i64
    %90 = llvm.add %87, %89  : i64
    %91 = llvm.mlir.constant(10 : index) : i64
    %92 = llvm.mul %56, %91  : i64
    %93 = llvm.add %90, %92  : i64
    %94 = llvm.add %93, %58  : i64
    %95 = llvm.getelementptr %85[%94] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %84, %95 : !llvm.ptr<f32>
    %96 = llvm.add %58, %14  : i64
    llvm.br ^bb8(%96 : i64)
  ^bb10:  // pred: ^bb8
    %97 = llvm.add %56, %14  : i64
    llvm.br ^bb6(%97 : i64)
  ^bb11:  // pred: ^bb6
    %98 = llvm.add %54, %14  : i64
    llvm.br ^bb4(%98 : i64)
  ^bb12:  // pred: ^bb4
    %99 = llvm.add %52, %14  : i64
    llvm.br ^bb2(%99 : i64)
  ^bb13:  // pred: ^bb2
    llvm.return
  ^bb14:  // pred: ^bb0
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
}
