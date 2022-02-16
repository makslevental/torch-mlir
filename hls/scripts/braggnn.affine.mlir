module attributes {torch.debug_module_name = "Conv2d"}  {
  memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_1x1x3x3xf32 : memref<1x1x3x3xf32> = dense<1.000000e+00>
  func @forward(%arg0: memref<1x1x8x8xf32>, %arg1: memref<1x1x6x6xf32>) {
    %true = arith.constant true
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = memref.get_global @__constant_1x1x3x3xf32 : memref<1x1x3x3xf32>
    %1 = memref.get_global @__constant_1xf32 : memref<1xf32>
    assert %true, "expect groups to be 1"
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c1 step %c1 {
        scf.for %arg4 = %c0 to %c6 step %c1 {
          scf.for %arg5 = %c0 to %c6 step %c1 {
            %2 = memref.load %1[%arg3] : memref<1xf32>
            memref.store %2, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x6x6xf32>
          }
        }
      }
    }
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c1 step %c1 {
        scf.for %arg4 = %c0 to %c6 step %c1 {
          scf.for %arg5 = %c0 to %c6 step %c1 {
            scf.for %arg6 = %c0 to %c1 step %c1 {
              scf.for %arg7 = %c0 to %c3 step %c1 {
                scf.for %arg8 = %c0 to %c3 step %c1 {
                  %2 = arith.addi %arg4, %arg7 : index
                  %3 = arith.addi %arg5, %arg8 : index
                  %4 = memref.load %arg0[%arg2, %arg6, %2, %3] : memref<1x1x8x8xf32>
                  %5 = memref.load %0[%arg3, %arg6, %arg7, %arg8] : memref<1x1x3x3xf32>
                  %6 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x6x6xf32>
                  %7 = arith.mulf %4, %5 : f32
                  %8 = arith.addf %6, %7 : f32
                  memref.store %8, %arg1[%arg2, %arg3, %arg4, %arg5] : memref<1x1x6x6xf32>
                }
              }
            }
          }
        }
      }
    }
    return
  }
}
