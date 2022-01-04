#loc0 = loc(unknown)
#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "BraggNN"}  {
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_2x8xf32 : memref<2x8xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_16xf32 : memref<16xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_16x32xf32 : memref<16x32xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_32xf32 : memref<32xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_32x64xf32 : memref<32x64xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_64x200xf32 : memref<64x200xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32> = dense<0.000000e+00> loc(#loc0)
  memref.global "private" constant @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32> = dense<0.000000e+00> loc(#loc0)
  func @forward(%arg0: memref<1x1x11x11xf32> loc(unknown), %arg1: memref<195008xi8> loc(unknown)) {
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %cst_0 = arith.constant 1.000000e-02 : f64 loc(#loc3)
    %true = arith.constant true loc(#loc4)
    %c0 = arith.constant 0 : index loc(#loc5)
    %c64 = arith.constant 64 : index loc(#loc5)
    %c1 = arith.constant 1 : index loc(#loc5)
    %c9 = arith.constant 9 : index loc(#loc6)
    %c23168 = arith.constant 23168 : index loc(#loc4)
    %c3 = arith.constant 3 : index loc(#loc2)
    %c30944 = arith.constant 30944 : index loc(#loc7)
    %c32 = arith.constant 32 : index loc(#loc5)
    %c43904 = arith.constant 43904 : index loc(#loc7)
    %c38720 = arith.constant 38720 : index loc(#loc8)
    %c41312 = arith.constant 41312 : index loc(#loc8)
    %c28352 = arith.constant 28352 : index loc(#loc9)
    %c36128 = arith.constant 36128 : index loc(#loc9)
    %c25760 = arith.constant 25760 : index loc(#loc10)
    %c48704 = arith.constant 48704 : index loc(#loc11)
    %c11936 = arith.constant 11936 : index loc(#loc11)
    %c33536 = arith.constant 33536 : index loc(#loc12)
    %c5184 = arith.constant 5184 : index loc(#loc13)
    %c17984 = arith.constant 17984 : index loc(#loc13)
    %c12800 = arith.constant 12800 : index loc(#loc6)
    %c10368 = arith.constant 10368 : index loc(#loc2)
    %c7 = arith.constant 7 : index loc(#loc6)
    %c46496 = arith.constant 46496 : index loc(#loc2)
    %c40768 = arith.constant 40768 : index loc(#loc6)
    %c42336 = arith.constant 42336 : index loc(#loc2)
    %c8 = arith.constant 8 : index loc(#loc5)
    %c5 = arith.constant 5 : index loc(#loc6)
    %c42536 = arith.constant 42536 : index loc(#loc2)
    %c200 = arith.constant 200 : index loc(#loc5)
    %c40832 = arith.constant 40832 : index loc(#loc5)
    %c40864 = arith.constant 40864 : index loc(#loc5)
    %c40896 = arith.constant 40896 : index loc(#loc1)
    %c40928 = arith.constant 40928 : index loc(#loc5)
    %c16 = arith.constant 16 : index loc(#loc5)
    %c48064 = arith.constant 48064 : index loc(#loc5)
    %c40944 = arith.constant 40944 : index loc(#loc5)
    %c40960 = arith.constant 40960 : index loc(#loc1)
    %c40976 = arith.constant 40976 : index loc(#loc5)
    %c48576 = arith.constant 48576 : index loc(#loc5)
    %c40984 = arith.constant 40984 : index loc(#loc5)
    %c40992 = arith.constant 40992 : index loc(#loc1)
    %c41000 = arith.constant 41000 : index loc(#loc5)
    %c2 = arith.constant 2 : index loc(#loc5)
    %c48736 = arith.constant 48736 : index loc(#loc5)
    %c41002 = arith.constant 41002 : index loc(#loc5)
    %0 = memref.get_global @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32> loc(#loc0)
    %1 = memref.get_global @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32> loc(#loc0)
    %2 = memref.get_global @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32> loc(#loc0)
    %3 = memref.get_global @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32> loc(#loc0)
    %4 = memref.get_global @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32> loc(#loc0)
    %5 = memref.get_global @__constant_64x200xf32 : memref<64x200xf32> loc(#loc0)
    %6 = memref.get_global @__constant_64xf32 : memref<64xf32> loc(#loc0)
    %7 = memref.get_global @__constant_32x64xf32 : memref<32x64xf32> loc(#loc0)
    %8 = memref.get_global @__constant_32xf32 : memref<32xf32> loc(#loc0)
    %9 = memref.get_global @__constant_16x32xf32 : memref<16x32xf32> loc(#loc0)
    %10 = memref.get_global @__constant_16xf32 : memref<16xf32> loc(#loc0)
    %11 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32> loc(#loc0)
    %12 = memref.get_global @__constant_8xf32 : memref<8xf32> loc(#loc0)
    %13 = memref.get_global @__constant_2x8xf32 : memref<2x8xf32> loc(#loc0)
    %14 = memref.get_global @__constant_2xf32 : memref<2xf32> loc(#loc0)
    assert %true, "expect groups to be 1" loc(#loc4)
    %15 = memref.view %arg1[%c0][] : memref<195008xi8> to memref<1x64x9x9xf32> loc(#loc4)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %6[%arg3] : memref<64xf32> loc(#loc4)
            memref.store %59, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc4)
          } loc(#loc4)
        } loc(#loc4)
      } loc(#loc4)
    } loc(#loc4)
    %16 = memref.view %arg1[%c23168][] : memref<195008xi8> to memref<1x64x9x9xf32> loc(#loc4)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc4)
            memref.store %59, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc4)
          } loc(#loc4)
        } loc(#loc4)
      } loc(#loc4)
    } loc(#loc4)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c1 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc4)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc4)
                  %61 = memref.load %arg0[%arg2, %arg6, %59, %60] : memref<1x1x11x11xf32> loc(#loc4)
                  %62 = memref.load %0[%arg3, %arg6, %arg7, %arg8] : memref<64x1x3x3xf32> loc(#loc4)
                  %63 = memref.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc4)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc4)
                } loc(#loc4)
              } loc(#loc4)
            } loc(#loc4)
          } loc(#loc4)
        } loc(#loc4)
      } loc(#loc4)
    } loc(#loc4)
    assert %true, "expect groups to be 1" loc(#loc7)
    %17 = memref.view %arg1[%c30944][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc7)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %8[%arg3] : memref<32xf32> loc(#loc7)
            memref.store %59, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc7)
          } loc(#loc7)
        } loc(#loc7)
      } loc(#loc7)
    } loc(#loc7)
    %18 = memref.view %arg1[%c43904][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc7)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc7)
            memref.store %59, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc7)
          } loc(#loc7)
        } loc(#loc7)
      } loc(#loc7)
    } loc(#loc7)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc7)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc7)
                  %61 = memref.load %16[%arg2, %arg6, %59, %60] : memref<1x64x9x9xf32> loc(#loc7)
                  %62 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32> loc(#loc7)
                  %63 = memref.load %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc7)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc7)
                } loc(#loc7)
              } loc(#loc7)
            } loc(#loc7)
          } loc(#loc7)
        } loc(#loc7)
      } loc(#loc7)
    } loc(#loc7)
    assert %true, "expect groups to be 1" loc(#loc8)
    %19 = memref.view %arg1[%c38720][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc8)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %8[%arg3] : memref<32xf32> loc(#loc8)
            memref.store %59, %19[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc8)
          } loc(#loc8)
        } loc(#loc8)
      } loc(#loc8)
    } loc(#loc8)
    %20 = memref.view %arg1[%c41312][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc8)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %19[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc8)
            memref.store %59, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc8)
          } loc(#loc8)
        } loc(#loc8)
      } loc(#loc8)
    } loc(#loc8)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc8)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc8)
                  %61 = memref.load %16[%arg2, %arg6, %59, %60] : memref<1x64x9x9xf32> loc(#loc8)
                  %62 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32> loc(#loc8)
                  %63 = memref.load %20[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc8)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %20[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc8)
                } loc(#loc8)
              } loc(#loc8)
            } loc(#loc8)
          } loc(#loc8)
        } loc(#loc8)
      } loc(#loc8)
    } loc(#loc8)
    assert %true, "expect groups to be 1" loc(#loc9)
    %21 = memref.view %arg1[%c28352][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc9)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %8[%arg3] : memref<32xf32> loc(#loc9)
            memref.store %59, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc9)
          } loc(#loc9)
        } loc(#loc9)
      } loc(#loc9)
    } loc(#loc9)
    %22 = memref.view %arg1[%c36128][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc9)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %21[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc9)
            memref.store %59, %22[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc9)
          } loc(#loc9)
        } loc(#loc9)
      } loc(#loc9)
    } loc(#loc9)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc9)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc9)
                  %61 = memref.load %16[%arg2, %arg6, %59, %60] : memref<1x64x9x9xf32> loc(#loc9)
                  %62 = memref.load %1[%arg3, %arg6, %arg7, %arg8] : memref<32x64x1x1xf32> loc(#loc9)
                  %63 = memref.load %22[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc9)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %22[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc9)
                } loc(#loc9)
              } loc(#loc9)
            } loc(#loc9)
          } loc(#loc9)
        } loc(#loc9)
      } loc(#loc9)
    } loc(#loc9)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %18[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc14)
            %60 = memref.load %20[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc14)
            %61 = arith.mulf %59, %60 : f32 loc(#loc14)
            memref.store %61, %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc14)
          } loc(#loc14)
        } loc(#loc14)
      } loc(#loc14)
    } loc(#loc14)
    %23 = memref.view %arg1[%c25760][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc10)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc10)
            %60 = math.exp %59 : f32 loc(#loc10)
            memref.store %60, %23[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc10)
          } loc(#loc10)
        } loc(#loc10)
      } loc(#loc10)
    } loc(#loc10)
    %24 = memref.view %arg1[%c23168][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc11)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %17[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc11)
            %60 = math.exp %59 : f32 loc(#loc11)
            memref.store %60, %24[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc11)
          } loc(#loc11)
        } loc(#loc11)
      } loc(#loc11)
    } loc(#loc11)
    %25 = memref.view %arg1[%c48704][] : memref<195008xi8> to memref<1x32x1x1xf32> loc(#loc11)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c1 step %c1 {
          scf.for %arg5 = %c0 to %c1 step %c1 {
            memref.store %cst, %25[%arg2, %arg3, %arg4, %arg5] : memref<1x32x1x1xf32> loc(#loc11)
          } loc(#loc11)
        } loc(#loc11)
      } loc(#loc11)
    } loc(#loc11)
    %26 = memref.view %arg1[%c11936][] : memref<195008xi8> to memref<1x32x1x1xf32> loc(#loc11)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c1 step %c1 {
          scf.for %arg5 = %c0 to %c1 step %c1 {
            %59 = memref.load %25[%arg2, %arg3, %arg4, %arg5] : memref<1x32x1x1xf32> loc(#loc11)
            memref.store %59, %26[%arg2, %arg3, %arg4, %arg5] : memref<1x32x1x1xf32> loc(#loc11)
          } loc(#loc11)
        } loc(#loc11)
      } loc(#loc11)
    } loc(#loc11)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %24[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc11)
            %60 = memref.load %26[%arg2, %arg3, %c0, %c0] : memref<1x32x1x1xf32> loc(#loc11)
            %61 = arith.addf %59, %60 : f32 loc(#loc11)
            memref.store %61, %26[%arg2, %arg3, %c0, %c0] : memref<1x32x1x1xf32> loc(#loc11)
          } loc(#loc11)
        } loc(#loc11)
      } loc(#loc11)
    } loc(#loc11)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %23[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc10)
            %60 = memref.load %26[%arg2, %arg3, %c0, %c0] : memref<1x32x1x1xf32> loc(#loc10)
            %61 = arith.divf %59, %60 : f32 loc(#loc10)
            memref.store %61, %21[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc10)
          } loc(#loc10)
        } loc(#loc10)
      } loc(#loc10)
    } loc(#loc10)
    %27 = memref.view %arg1[%c33536][] : memref<195008xi8> to memref<1x32x9x9xf32> loc(#loc12)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %21[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc12)
            %60 = memref.load %22[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc12)
            %61 = arith.mulf %59, %60 : f32 loc(#loc12)
            memref.store %61, %27[%arg2, %arg3, %arg4, %arg5] : memref<1x32x9x9xf32> loc(#loc12)
          } loc(#loc12)
        } loc(#loc12)
      } loc(#loc12)
    } loc(#loc12)
    assert %true, "expect groups to be 1" loc(#loc13)
    %28 = memref.view %arg1[%c5184][] : memref<195008xi8> to memref<1x64x9x9xf32> loc(#loc13)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %6[%arg3] : memref<64xf32> loc(#loc13)
            memref.store %59, %28[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc13)
          } loc(#loc13)
        } loc(#loc13)
      } loc(#loc13)
    } loc(#loc13)
    %29 = memref.view %arg1[%c17984][] : memref<195008xi8> to memref<1x64x9x9xf32> loc(#loc13)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %28[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc13)
            memref.store %59, %29[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc13)
          } loc(#loc13)
        } loc(#loc13)
      } loc(#loc13)
    } loc(#loc13)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              scf.for %arg7 = %c0 to %c1 step %c1 {
                scf.for %arg8 = %c0 to %c1 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc13)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc13)
                  %61 = memref.load %27[%arg2, %arg6, %59, %60] : memref<1x32x9x9xf32> loc(#loc13)
                  %62 = memref.load %2[%arg3, %arg6, %arg7, %arg8] : memref<64x32x1x1xf32> loc(#loc13)
                  %63 = memref.load %29[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc13)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %29[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc13)
                } loc(#loc13)
              } loc(#loc13)
            } loc(#loc13)
          } loc(#loc13)
        } loc(#loc13)
      } loc(#loc13)
    } loc(#loc13)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %29[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc15)
            %60 = memref.load %16[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc15)
            %61 = arith.sitofp %c1_i64 : i64 to f32 loc(#loc15)
            %62 = arith.mulf %60, %61 : f32 loc(#loc15)
            %63 = arith.addf %59, %62 : f32 loc(#loc15)
            memref.store %63, %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc15)
          } loc(#loc15)
        } loc(#loc15)
      } loc(#loc15)
    } loc(#loc15)
    %30 = memref.view %arg1[%c12800][] : memref<195008xi8> to memref<1x64x9x9xf32> loc(#loc6)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %c9 step %c1 {
          scf.for %arg5 = %c0 to %c9 step %c1 {
            %59 = memref.load %15[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc6)
            %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc6)
            %61 = select %60, %59, %cst : f32 loc(#loc6)
            %62 = select %60, %cst, %59 : f32 loc(#loc6)
            %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc6)
            %64 = arith.mulf %62, %63 : f32 loc(#loc6)
            %65 = arith.addf %61, %64 : f32 loc(#loc6)
            memref.store %65, %30[%arg2, %arg3, %arg4, %arg5] : memref<1x64x9x9xf32> loc(#loc6)
          } loc(#loc6)
        } loc(#loc6)
      } loc(#loc6)
    } loc(#loc6)
    assert %true, "expect groups to be 1" loc(#loc2)
    %31 = memref.view %arg1[%c10368][] : memref<195008xi8> to memref<1x32x7x7xf32> loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %59 = memref.load %8[%arg3] : memref<32xf32> loc(#loc2)
            memref.store %59, %31[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    %32 = memref.view %arg1[%c46496][] : memref<195008xi8> to memref<1x32x7x7xf32> loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %59 = memref.load %31[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc2)
            memref.store %59, %32[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc2)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc2)
                  %61 = memref.load %30[%arg2, %arg6, %59, %60] : memref<1x64x9x9xf32> loc(#loc2)
                  %62 = memref.load %3[%arg3, %arg6, %arg7, %arg8] : memref<32x64x3x3xf32> loc(#loc2)
                  %63 = memref.load %32[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc2)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %32[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc2)
                } loc(#loc2)
              } loc(#loc2)
            } loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    %33 = memref.view %arg1[%c40768][] : memref<195008xi8> to memref<1x32x7x7xf32> loc(#loc6)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c7 step %c1 {
          scf.for %arg5 = %c0 to %c7 step %c1 {
            %59 = memref.load %32[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc6)
            %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc6)
            %61 = select %60, %59, %cst : f32 loc(#loc6)
            %62 = select %60, %cst, %59 : f32 loc(#loc6)
            %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc6)
            %64 = arith.mulf %62, %63 : f32 loc(#loc6)
            %65 = arith.addf %61, %64 : f32 loc(#loc6)
            memref.store %65, %33[%arg2, %arg3, %arg4, %arg5] : memref<1x32x7x7xf32> loc(#loc6)
          } loc(#loc6)
        } loc(#loc6)
      } loc(#loc6)
    } loc(#loc6)
    assert %true, "expect groups to be 1" loc(#loc2)
    %34 = memref.view %arg1[%c42336][] : memref<195008xi8> to memref<1x8x5x5xf32> loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            %59 = memref.load %12[%arg3] : memref<8xf32> loc(#loc2)
            memref.store %59, %34[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    %35 = memref.view %arg1[%c42536][] : memref<195008xi8> to memref<1x8x5x5xf32> loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            %59 = memref.load %34[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc2)
            memref.store %59, %35[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %59 = affine.apply #map(%arg4, %arg7) loc(#loc2)
                  %60 = affine.apply #map(%arg5, %arg8) loc(#loc2)
                  %61 = memref.load %33[%arg2, %arg6, %59, %60] : memref<1x32x7x7xf32> loc(#loc2)
                  %62 = memref.load %4[%arg3, %arg6, %arg7, %arg8] : memref<8x32x3x3xf32> loc(#loc2)
                  %63 = memref.load %35[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc2)
                  %64 = arith.mulf %61, %62 : f32 loc(#loc0)
                  %65 = arith.addf %63, %64 : f32 loc(#loc0)
                  memref.store %65, %35[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc2)
                } loc(#loc2)
              } loc(#loc2)
            } loc(#loc2)
          } loc(#loc2)
        } loc(#loc2)
      } loc(#loc2)
    } loc(#loc2)
    %36 = memref.view %arg1[%c40768][] : memref<195008xi8> to memref<1x8x5x5xf32> loc(#loc6)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c5 step %c1 {
          scf.for %arg5 = %c0 to %c5 step %c1 {
            %59 = memref.load %35[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc6)
            %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc6)
            %61 = select %60, %59, %cst : f32 loc(#loc6)
            %62 = select %60, %cst, %59 : f32 loc(#loc6)
            %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc6)
            %64 = arith.mulf %62, %63 : f32 loc(#loc6)
            %65 = arith.addf %61, %64 : f32 loc(#loc6)
            memref.store %65, %36[%arg2, %arg3, %arg4, %arg5] : memref<1x8x5x5xf32> loc(#loc6)
          } loc(#loc6)
        } loc(#loc6)
      } loc(#loc6)
    } loc(#loc6)
    %37 = memref.cast %36 : memref<1x8x5x5xf32> to memref<?x?x?x?xf32> loc(#loc6)
    %38 = memref.collapse_shape %37 [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32> loc(#loc16)
    assert %true, "mismatching contracting dimension for aten.linear" loc(#loc5)
    %39 = memref.view %arg1[%c40768][] : memref<195008xi8> to memref<1x64xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %59 = memref.load %6[%arg3] : memref<64xf32> loc(#loc5)
        memref.store %59, %39[%arg2, %arg3] : memref<1x64xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %40 = memref.view %arg1[%c0][] : memref<195008xi8> to memref<200x64xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c200 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %59 = memref.load %5[%arg3, %arg2] : memref<64x200xf32> loc(#loc5)
        memref.store %59, %40[%arg2, %arg3] : memref<200x64xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %41 = memref.view %arg1[%c40832][] : memref<195008xi8> to memref<1x64xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %59 = memref.load %39[%arg2, %arg3] : memref<1x64xf32> loc(#loc5)
        memref.store %59, %41[%arg2, %arg3] : memref<1x64xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %42 = memref.dim %38, %c0 : memref<?x?xf32> loc(#loc5)
    %43 = memref.dim %38, %c1 : memref<?x?xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %42 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        scf.for %arg4 = %c0 to %43 step %c1 {
          %59 = memref.load %38[%arg2, %arg4] : memref<?x?xf32> loc(#loc5)
          %60 = memref.load %40[%arg4, %arg3] : memref<200x64xf32> loc(#loc5)
          %61 = memref.load %41[%arg2, %arg3] : memref<1x64xf32> loc(#loc5)
          %62 = arith.mulf %59, %60 : f32 loc(#loc0)
          %63 = arith.addf %61, %62 : f32 loc(#loc0)
          memref.store %63, %41[%arg2, %arg3] : memref<1x64xf32> loc(#loc5)
        } loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %59 = memref.load %41[%arg2, %arg3] : memref<1x64xf32> loc(#loc1)
        %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc1)
        %61 = select %60, %59, %cst : f32 loc(#loc1)
        %62 = select %60, %cst, %59 : f32 loc(#loc1)
        %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
        %64 = arith.mulf %62, %63 : f32 loc(#loc1)
        %65 = arith.addf %61, %64 : f32 loc(#loc1)
        memref.store %65, %39[%arg2, %arg3] : memref<1x64xf32> loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    %44 = memref.view %arg1[%c40832][] : memref<195008xi8> to memref<1x32xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %59 = memref.load %8[%arg3] : memref<32xf32> loc(#loc5)
        memref.store %59, %44[%arg2, %arg3] : memref<1x32xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %45 = memref.view %arg1[%c38720][] : memref<195008xi8> to memref<64x32xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c64 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %59 = memref.load %7[%arg3, %arg2] : memref<32x64xf32> loc(#loc5)
        memref.store %59, %45[%arg2, %arg3] : memref<64x32xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %46 = memref.view %arg1[%c40864][] : memref<195008xi8> to memref<1x32xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %59 = memref.load %44[%arg2, %arg3] : memref<1x32xf32> loc(#loc5)
        memref.store %59, %46[%arg2, %arg3] : memref<1x32xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c64 step %c1 {
          %59 = memref.load %39[%arg2, %arg4] : memref<1x64xf32> loc(#loc5)
          %60 = memref.load %45[%arg4, %arg3] : memref<64x32xf32> loc(#loc5)
          %61 = memref.load %46[%arg2, %arg3] : memref<1x32xf32> loc(#loc5)
          %62 = arith.mulf %59, %60 : f32 loc(#loc0)
          %63 = arith.addf %61, %62 : f32 loc(#loc0)
          memref.store %63, %46[%arg2, %arg3] : memref<1x32xf32> loc(#loc5)
        } loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %47 = memref.view %arg1[%c40896][] : memref<195008xi8> to memref<1x32xf32> loc(#loc1)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        %59 = memref.load %46[%arg2, %arg3] : memref<1x32xf32> loc(#loc1)
        %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc1)
        %61 = select %60, %59, %cst : f32 loc(#loc1)
        %62 = select %60, %cst, %59 : f32 loc(#loc1)
        %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
        %64 = arith.mulf %62, %63 : f32 loc(#loc1)
        %65 = arith.addf %61, %64 : f32 loc(#loc1)
        memref.store %65, %47[%arg2, %arg3] : memref<1x32xf32> loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    %48 = memref.view %arg1[%c40928][] : memref<195008xi8> to memref<1x16xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %59 = memref.load %10[%arg3] : memref<16xf32> loc(#loc5)
        memref.store %59, %48[%arg2, %arg3] : memref<1x16xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %49 = memref.view %arg1[%c48064][] : memref<195008xi8> to memref<32x16xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c32 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %59 = memref.load %9[%arg3, %arg2] : memref<16x32xf32> loc(#loc5)
        memref.store %59, %49[%arg2, %arg3] : memref<32x16xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %50 = memref.view %arg1[%c40944][] : memref<195008xi8> to memref<1x16xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %59 = memref.load %48[%arg2, %arg3] : memref<1x16xf32> loc(#loc5)
        memref.store %59, %50[%arg2, %arg3] : memref<1x16xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        scf.for %arg4 = %c0 to %c32 step %c1 {
          %59 = memref.load %47[%arg2, %arg4] : memref<1x32xf32> loc(#loc5)
          %60 = memref.load %49[%arg4, %arg3] : memref<32x16xf32> loc(#loc5)
          %61 = memref.load %50[%arg2, %arg3] : memref<1x16xf32> loc(#loc5)
          %62 = arith.mulf %59, %60 : f32 loc(#loc0)
          %63 = arith.addf %61, %62 : f32 loc(#loc0)
          memref.store %63, %50[%arg2, %arg3] : memref<1x16xf32> loc(#loc5)
        } loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %51 = memref.view %arg1[%c40960][] : memref<195008xi8> to memref<1x16xf32> loc(#loc1)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %59 = memref.load %50[%arg2, %arg3] : memref<1x16xf32> loc(#loc1)
        %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc1)
        %61 = select %60, %59, %cst : f32 loc(#loc1)
        %62 = select %60, %cst, %59 : f32 loc(#loc1)
        %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
        %64 = arith.mulf %62, %63 : f32 loc(#loc1)
        %65 = arith.addf %61, %64 : f32 loc(#loc1)
        memref.store %65, %51[%arg2, %arg3] : memref<1x16xf32> loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    %52 = memref.view %arg1[%c40976][] : memref<195008xi8> to memref<1x8xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %59 = memref.load %12[%arg3] : memref<8xf32> loc(#loc5)
        memref.store %59, %52[%arg2, %arg3] : memref<1x8xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %53 = memref.view %arg1[%c48576][] : memref<195008xi8> to memref<16x8xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c16 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %59 = memref.load %11[%arg3, %arg2] : memref<8x16xf32> loc(#loc5)
        memref.store %59, %53[%arg2, %arg3] : memref<16x8xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %54 = memref.view %arg1[%c40984][] : memref<195008xi8> to memref<1x8xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %59 = memref.load %52[%arg2, %arg3] : memref<1x8xf32> loc(#loc5)
        memref.store %59, %54[%arg2, %arg3] : memref<1x8xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c16 step %c1 {
          %59 = memref.load %51[%arg2, %arg4] : memref<1x16xf32> loc(#loc5)
          %60 = memref.load %53[%arg4, %arg3] : memref<16x8xf32> loc(#loc5)
          %61 = memref.load %54[%arg2, %arg3] : memref<1x8xf32> loc(#loc5)
          %62 = arith.mulf %59, %60 : f32 loc(#loc0)
          %63 = arith.addf %61, %62 : f32 loc(#loc0)
          memref.store %63, %54[%arg2, %arg3] : memref<1x8xf32> loc(#loc5)
        } loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %55 = memref.view %arg1[%c40992][] : memref<195008xi8> to memref<1x8xf32> loc(#loc1)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        %59 = memref.load %54[%arg2, %arg3] : memref<1x8xf32> loc(#loc1)
        %60 = arith.cmpf ugt, %59, %cst : f32 loc(#loc1)
        %61 = select %60, %59, %cst : f32 loc(#loc1)
        %62 = select %60, %cst, %59 : f32 loc(#loc1)
        %63 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
        %64 = arith.mulf %62, %63 : f32 loc(#loc1)
        %65 = arith.addf %61, %64 : f32 loc(#loc1)
        memref.store %65, %55[%arg2, %arg3] : memref<1x8xf32> loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    %56 = memref.view %arg1[%c41000][] : memref<195008xi8> to memref<1x2xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %59 = memref.load %14[%arg3] : memref<2xf32> loc(#loc5)
        memref.store %59, %56[%arg2, %arg3] : memref<1x2xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %57 = memref.view %arg1[%c48736][] : memref<195008xi8> to memref<8x2xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %59 = memref.load %13[%arg3, %arg2] : memref<2x8xf32> loc(#loc5)
        memref.store %59, %57[%arg2, %arg3] : memref<8x2xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    %58 = memref.view %arg1[%c41002][] : memref<195008xi8> to memref<1x2xf32> loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %59 = memref.load %56[%arg2, %arg3] : memref<1x2xf32> loc(#loc5)
        memref.store %59, %58[%arg2, %arg3] : memref<1x2xf32> loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c2 step %c1 {
        scf.for %arg4 = %c0 to %c8 step %c1 {
          %59 = memref.load %55[%arg2, %arg4] : memref<1x8xf32> loc(#loc5)
          %60 = memref.load %57[%arg4, %arg3] : memref<8x2xf32> loc(#loc5)
          %61 = memref.load %58[%arg2, %arg3] : memref<1x2xf32> loc(#loc5)
          %62 = arith.mulf %59, %60 : f32 loc(#loc0)
          %63 = arith.addf %61, %62 : f32 loc(#loc0)
          memref.store %63, %58[%arg2, %arg3] : memref<1x2xf32> loc(#loc5)
        } loc(#loc5)
      } loc(#loc5)
    } loc(#loc5)
    return loc(#loc0)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1578:17 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":105:15))
#loc2 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc3 = loc(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:35 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc4 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":101:15))
#loc5 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1951:11 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/linear.py":103:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":105:15))
#loc6 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1578:17 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc7 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":35:16) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc8 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":36:14) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc9 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":37:12) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc10 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":40:20 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc11 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":40:38 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc12 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":41:22 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc13 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":43:19) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc14 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":39:20 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc15 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":44:19 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc16 = loc("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":104:15)
