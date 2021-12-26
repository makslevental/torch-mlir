module attributes {llvm.data_layout = "", torch.debug_module_name = "Conv2dNoPaddingOutModule"}  {
  llvm.mlir.global private constant @__constant_10x2x3x3xf32(dense<"0x004CE7BABD79013E42A646BE27A031BEC8EBB9BDC671813DE00699BB015F3F3E805AABBC74777F3D3EE291BD7CC53DBD029566BE0BD91FBED4FCC6BD900D0F3CE2D5BE3D2BD2103E94A023BEB234D2BDDE54AF3DF56B483EE0AF46BDA39C343EC09C1BBD4850CC3C818B5A3ED6E65FBED9F117BE3C6A74BDDE29BCBDD388503EAE711CBE0836DEBD8D9F28BE2B0C62BE1EE40CBEC9784F3ED265D73DD6F5E93D60184B3C4F7BF7BD4C56233D115B61BEE0652EBE11DBF8BD5948183E93830D3E4815D6BD50570BBC235D1A3E6FF16F3E0E95BF3D6C6C023DF7D3213EE11C0EBE3CE7333D86203BBE5C4827BE435DF9BDBE6ADA3D5221C23D3DF80EBE26D5913DD97F043EB0B5F3BC9072133C3CB25F3DC9BB153E69C0673E3DFF39BED0E6B0BD8AB6BD3D5BFA473E4F08523E01F7543ECC22403DB7E151BE38A0B13CDCFE16BE85EF60BE1D72563E0F85373E31C370BEE4B3343D30A322BD94DF1EBDB4F7DCBDDEA1B93D36F50EBED6F5B03D521DF43D39C82C3E667EB43DB2E16EBEA4911CBE4A07F13D34114A3DC8473CBEB2FA0ABEF50E633ED1A1223E307AD2BD54FC72BD11EB65BEC0D28ABBF4C135BE8A2C3ABEE0C754BC9CF4103D10B0C5BD93370F3E60E012BE85005B3E4367253EB6884BBE304870BD90402E3CB4DB0C3D0CFA643DAA6EBD3DB051673C3888EBBDBE6AE43D028667BE1F0E0FBE74AD71BD7523EBBDF6DEA8BDBBD245BEEC5C4DBD445E4E3DBA3C1DBEB02E46BCF1C62C3E887EC6BC20A7D63B0893A6BCEC62433D3577193E63A0643ECB46193E4D26653ED8A48BBC6ED158BE81D8E4BDBB57243E0045C8BAEFEEEFBDD2F438BE5DE061BE86B54BBE74D343BDDF5C043E197D023E41C668BE358E163EA4DD3CBE6C1A4CBDA2BAC3BD34F539BD708E3DBD669558BE0B6650BE2C1217BD80C3473B6A49DBBDEED6B53DC03C59BE304F82BC8341543E9CE5C4BDB3F2593E76D1AE3D32D159BE0FB5183EE0CFDEBC1C7DD7BDED00413E880A43BE"> : tensor<10x2x3x3xf32>) : !llvm.array<10 x array<2 x array<3 x array<3 x f32>>>>
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
    %24 = llvm.mlir.constant(5 : index) : i64
    %25 = llvm.mlir.constant(2 : index) : i64
    %26 = llvm.mlir.constant(10 : index) : i64
    %27 = llvm.mlir.constant(3 : index) : i64
    %28 = llvm.mlir.constant(8 : index) : i64
    %29 = llvm.mlir.constant(18 : index) : i64
    %30 = llvm.mlir.constant(0 : index) : i64
    %31 = llvm.mlir.constant(1 : index) : i64
    %32 = llvm.mlir.constant(10 : index) : i64
    %33 = llvm.mlir.constant(2 : index) : i64
    %34 = llvm.mlir.constant(3 : index) : i64
    %35 = llvm.mlir.constant(3 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(9 : index) : i64
    %38 = llvm.mlir.constant(18 : index) : i64
    %39 = llvm.mlir.constant(180 : index) : i64
    %40 = llvm.mlir.null : !llvm.ptr<f32>
    %41 = llvm.getelementptr %40[%39] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %42 = llvm.ptrtoint %41 : !llvm.ptr<f32> to i64
    %43 = llvm.mlir.addressof @__constant_10x2x3x3xf32 : !llvm.ptr<array<10 x array<2 x array<3 x array<3 x f32>>>>>
    %44 = llvm.mlir.constant(0 : index) : i64
    %45 = llvm.getelementptr %43[%44, %44, %44, %44, %44] : (!llvm.ptr<array<10 x array<2 x array<3 x array<3 x f32>>>>>, i64, i64, i64, i64, i64) -> !llvm.ptr<f32>
    %46 = llvm.mlir.constant(3735928559 : index) : i64
    %47 = llvm.inttoptr %46 : i64 to !llvm.ptr<f32>
    %48 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %49 = llvm.insertvalue %47, %48[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %50 = llvm.insertvalue %45, %49[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %51 = llvm.mlir.constant(0 : index) : i64
    %52 = llvm.insertvalue %51, %50[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %53 = llvm.insertvalue %32, %52[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %54 = llvm.insertvalue %33, %53[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %55 = llvm.insertvalue %34, %54[3, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %56 = llvm.insertvalue %35, %55[3, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %57 = llvm.insertvalue %38, %56[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %58 = llvm.insertvalue %37, %57[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %59 = llvm.insertvalue %35, %58[4, 2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %60 = llvm.insertvalue %36, %59[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    llvm.br ^bb1(%30 : i64)
  ^bb1(%61: i64):  // 2 preds: ^bb0, ^bb20
    %62 = llvm.icmp "slt" %61, %24 : i64
    llvm.cond_br %62, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%30 : i64)
  ^bb3(%63: i64):  // 2 preds: ^bb2, ^bb19
    %64 = llvm.icmp "slt" %63, %26 : i64
    llvm.cond_br %64, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%30 : i64)
  ^bb5(%65: i64):  // 2 preds: ^bb4, ^bb18
    %66 = llvm.icmp "slt" %65, %28 : i64
    llvm.cond_br %66, ^bb6, ^bb19
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%30 : i64)
  ^bb7(%67: i64):  // 2 preds: ^bb6, ^bb17
    %68 = llvm.icmp "slt" %67, %29 : i64
    llvm.cond_br %68, ^bb8, ^bb18
  ^bb8:  // pred: ^bb7
    llvm.br ^bb9(%30 : i64)
  ^bb9(%69: i64):  // 2 preds: ^bb8, ^bb16
    %70 = llvm.icmp "slt" %69, %25 : i64
    llvm.cond_br %70, ^bb10, ^bb17
  ^bb10:  // pred: ^bb9
    llvm.br ^bb11(%30 : i64)
  ^bb11(%71: i64):  // 2 preds: ^bb10, ^bb15
    %72 = llvm.icmp "slt" %71, %27 : i64
    llvm.cond_br %72, ^bb12, ^bb16
  ^bb12:  // pred: ^bb11
    llvm.br ^bb13(%30 : i64)
  ^bb13(%73: i64):  // 2 preds: ^bb12, ^bb14
    %74 = llvm.icmp "slt" %73, %27 : i64
    llvm.cond_br %74, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %75 = llvm.add %65, %71  : i64
    %76 = llvm.add %67, %73  : i64
    %77 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %78 = llvm.mlir.constant(400 : index) : i64
    %79 = llvm.mul %61, %78  : i64
    %80 = llvm.mlir.constant(200 : index) : i64
    %81 = llvm.mul %69, %80  : i64
    %82 = llvm.add %79, %81  : i64
    %83 = llvm.mlir.constant(20 : index) : i64
    %84 = llvm.mul %75, %83  : i64
    %85 = llvm.add %82, %84  : i64
    %86 = llvm.add %85, %76  : i64
    %87 = llvm.getelementptr %77[%86] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %88 = llvm.load %87 : !llvm.ptr<f32>
    %89 = llvm.mlir.constant(18 : index) : i64
    %90 = llvm.mul %63, %89  : i64
    %91 = llvm.mlir.constant(9 : index) : i64
    %92 = llvm.mul %69, %91  : i64
    %93 = llvm.add %90, %92  : i64
    %94 = llvm.mlir.constant(3 : index) : i64
    %95 = llvm.mul %71, %94  : i64
    %96 = llvm.add %93, %95  : i64
    %97 = llvm.add %96, %73  : i64
    %98 = llvm.getelementptr %45[%97] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %99 = llvm.load %98 : !llvm.ptr<f32>
    %100 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %101 = llvm.mlir.constant(1440 : index) : i64
    %102 = llvm.mul %61, %101  : i64
    %103 = llvm.mlir.constant(144 : index) : i64
    %104 = llvm.mul %63, %103  : i64
    %105 = llvm.add %102, %104  : i64
    %106 = llvm.mlir.constant(18 : index) : i64
    %107 = llvm.mul %65, %106  : i64
    %108 = llvm.add %105, %107  : i64
    %109 = llvm.add %108, %67  : i64
    %110 = llvm.getelementptr %100[%109] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %111 = llvm.load %110 : !llvm.ptr<f32>
    %112 = llvm.fmul %88, %99  : f32
    %113 = llvm.fadd %111, %112  : f32
    %114 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
    %115 = llvm.mlir.constant(1440 : index) : i64
    %116 = llvm.mul %61, %115  : i64
    %117 = llvm.mlir.constant(144 : index) : i64
    %118 = llvm.mul %63, %117  : i64
    %119 = llvm.add %116, %118  : i64
    %120 = llvm.mlir.constant(18 : index) : i64
    %121 = llvm.mul %65, %120  : i64
    %122 = llvm.add %119, %121  : i64
    %123 = llvm.add %122, %67  : i64
    %124 = llvm.getelementptr %114[%123] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %113, %124 : !llvm.ptr<f32>
    %125 = llvm.add %73, %31  : i64
    llvm.br ^bb13(%125 : i64)
  ^bb15:  // pred: ^bb13
    %126 = llvm.add %71, %31  : i64
    llvm.br ^bb11(%126 : i64)
  ^bb16:  // pred: ^bb11
    %127 = llvm.add %69, %31  : i64
    llvm.br ^bb9(%127 : i64)
  ^bb17:  // pred: ^bb9
    %128 = llvm.add %67, %31  : i64
    llvm.br ^bb7(%128 : i64)
  ^bb18:  // pred: ^bb7
    %129 = llvm.add %65, %31  : i64
    llvm.br ^bb5(%129 : i64)
  ^bb19:  // pred: ^bb5
    %130 = llvm.add %63, %31  : i64
    llvm.br ^bb3(%130 : i64)
  ^bb20:  // pred: ^bb3
    %131 = llvm.add %61, %31  : i64
    llvm.br ^bb1(%131 : i64)
  ^bb21:  // pred: ^bb1
    llvm.return
  }
}
