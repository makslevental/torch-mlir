module attributes {torch.debug_module_name = "Conv2dNoPaddingOutModule"}  {
  memref.global "private" constant @__constant_10x2x3x3xf32 : memref<10x2x3x3xf32> = dense<"0x004CE7BABD79013E42A646BE27A031BEC8EBB9BDC671813DE00699BB015F3F3E805AABBC74777F3D3EE291BD7CC53DBD029566BE0BD91FBED4FCC6BD900D0F3CE2D5BE3D2BD2103E94A023BEB234D2BDDE54AF3DF56B483EE0AF46BDA39C343EC09C1BBD4850CC3C818B5A3ED6E65FBED9F117BE3C6A74BDDE29BCBDD388503EAE711CBE0836DEBD8D9F28BE2B0C62BE1EE40CBEC9784F3ED265D73DD6F5E93D60184B3C4F7BF7BD4C56233D115B61BEE0652EBE11DBF8BD5948183E93830D3E4815D6BD50570BBC235D1A3E6FF16F3E0E95BF3D6C6C023DF7D3213EE11C0EBE3CE7333D86203BBE5C4827BE435DF9BDBE6ADA3D5221C23D3DF80EBE26D5913DD97F043EB0B5F3BC9072133C3CB25F3DC9BB153E69C0673E3DFF39BED0E6B0BD8AB6BD3D5BFA473E4F08523E01F7543ECC22403DB7E151BE38A0B13CDCFE16BE85EF60BE1D72563E0F85373E31C370BEE4B3343D30A322BD94DF1EBDB4F7DCBDDEA1B93D36F50EBED6F5B03D521DF43D39C82C3E667EB43DB2E16EBEA4911CBE4A07F13D34114A3DC8473CBEB2FA0ABEF50E633ED1A1223E307AD2BD54FC72BD11EB65BEC0D28ABBF4C135BE8A2C3ABEE0C754BC9CF4103D10B0C5BD93370F3E60E012BE85005B3E4367253EB6884BBE304870BD90402E3CB4DB0C3D0CFA643DAA6EBD3DB051673C3888EBBDBE6AE43D028667BE1F0E0FBE74AD71BD7523EBBDF6DEA8BDBBD245BEEC5C4DBD445E4E3DBA3C1DBEB02E46BCF1C62C3E887EC6BC20A7D63B0893A6BCEC62433D3577193E63A0643ECB46193E4D26653ED8A48BBC6ED158BE81D8E4BDBB57243E0045C8BAEFEEEFBDD2F438BE5DE061BE86B54BBE74D343BDDF5C043E197D023E41C668BE358E163EA4DD3CBE6C1A4CBDA2BAC3BD34F539BD708E3DBD669558BE0B6650BE2C1217BD80C3473B6A49DBBDEED6B53DC03C59BE304F82BC8341543E9CE5C4BDB3F2593E76D1AE3D32D159BE0FB5183EE0CFDEBC1C7DD7BDED00413E880A43BE">
  func @forward(%arg0: memref<5x2x10x20xf32>, %arg1: memref<5x10x8x18xf32>) {
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %c3 = arith.constant 3 : index
    %c8 = arith.constant 8 : index
    %c18 = arith.constant 18 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @__constant_10x2x3x3xf32 : memref<10x2x3x3xf32>
    br ^bb1(%c0 : index)
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb20
    %2 = arith.cmpi slt, %1, %c5 : index
    cond_br %2, ^bb2, ^bb21
  ^bb2:  // pred: ^bb1
    br ^bb3(%c0 : index)
  ^bb3(%3: index):  // 2 preds: ^bb2, ^bb19
    %4 = arith.cmpi slt, %3, %c10 : index
    cond_br %4, ^bb4, ^bb20
  ^bb4:  // pred: ^bb3
    br ^bb5(%c0 : index)
  ^bb5(%5: index):  // 2 preds: ^bb4, ^bb18
    %6 = arith.cmpi slt, %5, %c8 : index
    cond_br %6, ^bb6, ^bb19
  ^bb6:  // pred: ^bb5
    br ^bb7(%c0 : index)
  ^bb7(%7: index):  // 2 preds: ^bb6, ^bb17
    %8 = arith.cmpi slt, %7, %c18 : index
    cond_br %8, ^bb8, ^bb18
  ^bb8:  // pred: ^bb7
    br ^bb9(%c0 : index)
  ^bb9(%9: index):  // 2 preds: ^bb8, ^bb16
    %10 = arith.cmpi slt, %9, %c2 : index
    cond_br %10, ^bb10, ^bb17
  ^bb10:  // pred: ^bb9
    br ^bb11(%c0 : index)
  ^bb11(%11: index):  // 2 preds: ^bb10, ^bb15
    %12 = arith.cmpi slt, %11, %c3 : index
    cond_br %12, ^bb12, ^bb16
  ^bb12:  // pred: ^bb11
    br ^bb13(%c0 : index)
  ^bb13(%13: index):  // 2 preds: ^bb12, ^bb14
    %14 = arith.cmpi slt, %13, %c3 : index
    cond_br %14, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    %15 = arith.addi %5, %11 : index
    %16 = arith.addi %7, %13 : index
    %17 = memref.load %arg0[%1, %9, %15, %16] : memref<5x2x10x20xf32>
    %18 = memref.load %0[%3, %9, %11, %13] : memref<10x2x3x3xf32>
    %19 = memref.load %arg1[%1, %3, %5, %7] : memref<5x10x8x18xf32>
    %20 = arith.mulf %17, %18 : f32
    %21 = arith.addf %19, %20 : f32
    memref.store %21, %arg1[%1, %3, %5, %7] : memref<5x10x8x18xf32>
    %22 = arith.addi %13, %c1 : index
    br ^bb13(%22 : index)
  ^bb15:  // pred: ^bb13
    %23 = arith.addi %11, %c1 : index
    br ^bb11(%23 : index)
  ^bb16:  // pred: ^bb11
    %24 = arith.addi %9, %c1 : index
    br ^bb9(%24 : index)
  ^bb17:  // pred: ^bb9
    %25 = arith.addi %7, %c1 : index
    br ^bb7(%25 : index)
  ^bb18:  // pred: ^bb7
    %26 = arith.addi %5, %c1 : index
    br ^bb5(%26 : index)
  ^bb19:  // pred: ^bb5
    %27 = arith.addi %3, %c1 : index
    br ^bb3(%27 : index)
  ^bb20:  // pred: ^bb3
    %28 = arith.addi %1, %c1 : index
    br ^bb1(%28 : index)
  ^bb21:  // pred: ^bb1
    return
  }
}
