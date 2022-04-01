from collections import defaultdict

print("""
#map = affine_map<(d0, d1) -> (d0 + d1)>
module attributes {torch.debug_module_name = "BraggNN"} {
  memref.global "private" constant @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64x200xf32 : memref<64x200xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_64xf32 : memref<64xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32x64xf32 : memref<32x64xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_32xf32 : memref<32xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16x32xf32 : memref<16x32xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_16xf32 : memref<16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8x16xf32 : memref<8x16xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_8xf32 : memref<8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2x8xf32 : memref<2x8xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_2xf32 : memref<2xf32> = dense<1.000000e+00>
  func @forward(%arg0: memref<1x1x11x11xf32>, %arg1: memref<1x2xf32>) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 1.000000e-02 : f64
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_64x1x3x3xf32 : memref<64x1x3x3xf32>
    %1 = memref.get_global @__constant_32x64x1x1xf32 : memref<32x64x1x1xf32>
    %2 = memref.get_global @__constant_64x32x1x1xf32 : memref<64x32x1x1xf32>
    %3 = memref.get_global @__constant_32x64x3x3xf32 : memref<32x64x3x3xf32>
    %4 = memref.get_global @__constant_8x32x3x3xf32 : memref<8x32x3x3xf32>
    %5 = memref.get_global @__constant_64x200xf32 : memref<64x200xf32>
    %6 = memref.get_global @__constant_64xf32 : memref<64xf32>
    %7 = memref.get_global @__constant_32x64xf32 : memref<32x64xf32>
    %8 = memref.get_global @__constant_32xf32 : memref<32xf32>
    %9 = memref.get_global @__constant_16x32xf32 : memref<16x32xf32>
    %10 = memref.get_global @__constant_16xf32 : memref<16xf32>
    %11 = memref.get_global @__constant_8x16xf32 : memref<8x16xf32>
    %12 = memref.get_global @__constant_8xf32 : memref<8xf32>
    %13 = memref.get_global @__constant_2x8xf32 : memref<2x8xf32>
    %14 = memref.get_global @__constant_2xf32 : memref<2xf32>
    %15 = memref.alloca() : memref<1x64x9x9xf32>
""")


def fun():
    first_conv_layer_size= 64
    val_id = 100
    for i in range(20):
        print(f"%cst_idx_{i} = arith.constant {i} : index")
    for arg2 in range(0, 1):
        for arg3 in range(0, first_conv_layer_size):
            for arg4 in range(0, 9):
                for arg5 in range(0, 9):
                    val_id += 1
                    print(f"""%{val_id} = affine.load %6[%cst_idx_{arg3}] : memref<64xf32>
affine.store %{val_id}, %15[%cst_idx_{arg2}, %cst_idx_{arg3}, %cst_idx_{arg4}, %cst_idx_{arg5}] : memref<1x64x9x9xf32>""")

    for arg2 in range(0, 1):
        for arg3 in range(0, first_conv_layer_size):
            for arg4 in range(0, 9):
                for arg5 in range(0, 9):
                    for arg6 in range(0, 1):
                        for arg7 in range(0, 3):
                            for arg8 in range(0, 3):
                                val_id += 10
                                print(f"""%{val_id+2} = affine.load %arg0[%cst_idx_{arg2}, %cst_idx_{arg6}, %cst_idx_{arg4 + arg7}, %cst_idx_{arg5 + arg8}] : memref<1x1x11x11xf32>
%{val_id+3} = affine.load %0[%cst_idx_{arg3}, %cst_idx_{arg6}, %cst_idx_{arg7}, %cst_idx_{arg8}] : memref<64x1x3x3xf32>
%{val_id+4} = affine.load %15[%cst_idx_{arg2}, %cst_idx_{arg3}, %cst_idx_{arg4}, %cst_idx_{arg5}] : memref<1x64x9x9xf32>
%{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
%{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
affine.store %{val_id+6}, %15[%cst_idx_{arg2}, %cst_idx_{arg3}, %cst_idx_{arg4}, %cst_idx_{arg5}] : memref<1x64x9x9xf32>""")

#     print("""%16 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %8[{arg3}] : memref<32xf32>
# affine.store %{val_id}, %16[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     for arg6 in range(0, 64):
#                         for arg7 in range(0, 1):
#                             for arg8 in range(0, 1):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %15[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x64x9x9xf32>
# %{val_id+3} = affine.load %1[{arg3}, {arg6}, {arg7}, {arg8}] : memref<32x64x1x1xf32>
# %{val_id+4} = affine.load %16[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %16[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#     print("""%17 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %8[{arg3}] : memref<32xf32>
# affine.store %{val_id}, %17[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     for arg6 in range(0, 64):
#                         for arg7 in range(0, 1):
#                             for arg8 in range(0, 1):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %15[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x64x9x9xf32>
# %{val_id+3} = affine.load %1[{arg3}, {arg6}, {arg7}, {arg8}] : memref<32x64x1x1xf32>
# %{val_id+4} = affine.load %17[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %17[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#     print("""%18 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %8[{arg3}] : memref<32xf32>
# affine.store %{val_id}, %18[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     for arg6 in range(0, 64):
#                         for arg7 in range(0, 1):
#                             for arg8 in range(0, 1):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %15[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x64x9x9xf32>
# %{val_id+3} = affine.load %1[{arg3}, {arg6}, {arg7}, {arg8}] : memref<32x64x1x1xf32>
# %{val_id+4} = affine.load %18[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %18[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%19 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %16[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %17[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+2} = arith.mulf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %19[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%20 = memref.alloca() : memref<1x32x9x1xi64>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 1):
#                     print(f"""affine.store %c0_i64, %20[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x1xi64>""")
#
#
#     print("""%21 = memref.alloca() : memref<1x32x9x1xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 1):
#                     print(f"""affine.store %cst_0, %21[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x1xf32>""")
#
#     i64_constants = {}
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     if arg5 not in i64_constants:
#                         print(f"%{val_id+3} = arith.constant {arg5} : i64")
#                         i64_constants[arg5] = f"%{val_id+3}"
#                     val_id3 = i64_constants[arg5]
#
#                     print(f"""%{val_id} = affine.load %19[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %21[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>
# %{val_id+2} = affine.load %20[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xi64>
# %{val_id+4} = arith.cmpf ogt, %{val_id}, %{val_id+1} : f32
# %{val_id+5} = arith.select %{val_id+4}, %{val_id}, %{val_id+1} : f32
# %{val_id+6} = arith.select %{val_id+4}, {val_id3}, %{val_id+2} : i64
# affine.store %{val_id+5}, %21[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>
# affine.store %{val_id+6}, %20[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xi64>""")
#
#
#     print("""%22 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %19[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %21[%c0, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>
# %{val_id+2} = arith.subf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %22[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%23 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %22[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = math.exp %{val_id} : f32
# affine.store %{val_id+1}, %23[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%24 = memref.alloca() : memref<1x32x9x1xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 1):
#                     print(f"""affine.store %cst, %24[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x1xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %23[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %24[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>
# %{val_id+2} = arith.addf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %24[{arg2}, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>""")
#
#
#     print("""%25 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %23[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %24[%c0, {arg3}, {arg4}, %c0] : memref<1x32x9x1xf32>
# %{val_id+2} = arith.divf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %25[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%26 = memref.alloca() : memref<1x32x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %25[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+1} = affine.load %18[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>
# %{val_id+2} = arith.mulf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %26[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x9x9xf32>""")
#
#
#     print("""%27 = memref.alloca() : memref<1x64x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %6[{arg3}] : memref<64xf32>
# affine.store %{val_id}, %27[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     for arg6 in range(0, 32):
#                         for arg7 in range(0, 1):
#                             for arg8 in range(0, 1):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %26[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x32x9x9xf32>
# %{val_id+3} = affine.load %2[{arg3}, {arg6}, {arg7}, {arg8}] : memref<64x32x1x1xf32>
# %{val_id+4} = affine.load %27[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %27[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>""")
#
#
#     print("""%28 = memref.alloca() : memref<1x64x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %27[%c0, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>
# %{val_id+1} = affine.load %15[%c0, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>
# %{val_id+2} = arith.addf %{val_id}, %{val_id+1} : f32
# affine.store %{val_id+2}, %28[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>""")
#
#
#     print("""%29 = memref.alloca() : memref<1x64x9x9xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             for arg4 in range(0, 9):
#                 for arg5 in range(0, 9):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %28[%c0, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %29[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x64x9x9xf32>""")
#
#
#     print("""%30 = memref.alloca() : memref<1x32x7x7xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 7):
#                 for arg5 in range(0, 7):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %8[{arg3}] : memref<32xf32>
# affine.store %{val_id}, %30[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x7x7xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 7):
#                 for arg5 in range(0, 7):
#                     for arg6 in range(0, 64):
#                         for arg7 in range(0, 3):
#                             for arg8 in range(0, 3):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %29[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x64x9x9xf32>
# %{val_id+3} = affine.load %3[{arg3}, {arg6}, {arg7}, {arg8}] : memref<32x64x3x3xf32>
# %{val_id+4} = affine.load %30[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x7x7xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %30[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x7x7xf32>""")
#
#
#     print("""%31 = memref.alloca() : memref<1x32x7x7xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 7):
#                 for arg5 in range(0, 7):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %30[%c0, {arg3}, {arg4}, {arg5}] : memref<1x32x7x7xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %31[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x32x7x7xf32>""")
#
#
#     print("""%32 = memref.alloca() : memref<1x8x5x5xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             for arg4 in range(0, 5):
#                 for arg5 in range(0, 5):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %12[{arg3}] : memref<8xf32>
# affine.store %{val_id}, %32[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x8x5x5xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             for arg4 in range(0, 5):
#                 for arg5 in range(0, 5):
#                     for arg6 in range(0, 32):
#                         for arg7 in range(0, 3):
#                             for arg8 in range(0, 3):
#                                 val_id += 10
#                                 print(f"""%{val_id+2} = affine.load %31[{arg2}, {arg6}, {arg4 + arg7}, {arg5 + arg8}] : memref<1x32x7x7xf32>
# %{val_id+3} = affine.load %4[{arg3}, {arg6}, {arg7}, {arg8}] : memref<8x32x3x3xf32>
# %{val_id+4} = affine.load %32[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x8x5x5xf32>
# %{val_id+5} = arith.mulf %{val_id+2}, %{val_id+3} : f32
# %{val_id+6} = arith.addf %{val_id+4}, %{val_id+5} : f32
# affine.store %{val_id+6}, %32[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x8x5x5xf32>""")
#
#
#     print("""%33 = memref.alloca() : memref<1x8x5x5xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             for arg4 in range(0, 5):
#                 for arg5 in range(0, 5):
#                     val_id += 10
#                     print(f"""%{val_id} = affine.load %32[%c0, {arg3}, {arg4}, {arg5}] : memref<1x8x5x5xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %33[{arg2}, {arg3}, {arg4}, {arg5}] : memref<1x8x5x5xf32>""")
#
#
#     print("""%34 = memref.collapse_shape %33 [[0], [1, 2, 3]] : memref<1x8x5x5xf32> into memref<1x200xf32>""")
#
#     print("""%35 = memref.alloca() : memref<1x64xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %6[{arg3}] : memref<64xf32>
# affine.store %{val_id}, %35[{arg2}, {arg3}] : memref<1x64xf32>""")
#
#
#     print("""%36 = memref.alloca() : memref<200x64xf32>""")
#     for arg2 in range(0, 200):
#         for arg3 in range(0, 64):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %5[{arg3}, {arg2}] : memref<64x200xf32>
# affine.store %{val_id}, %36[{arg2}, {arg3}] : memref<200x64xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             for arg4 in range(0, 200):
#                 val_id += 10
#                 print(f"""%{val_id} = affine.load %34[{arg2}, {arg4}] : memref<1x200xf32>
# %{val_id+1} = affine.load %36[{arg4}, {arg3}] : memref<200x64xf32>
# %{val_id+2} = affine.load %35[{arg2}, {arg3}] : memref<1x64xf32>
# %{val_id+3} = arith.mulf %{val_id}, %{val_id+1} : f32
# %{val_id+4} = arith.addf %{val_id+2}, %{val_id+3} : f32
# affine.store %{val_id+4}, %35[{arg2}, {arg3}] : memref<1x64xf32>""")
#
#
#     print("""%37 = memref.alloca() : memref<1x64xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 64):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %35[%c0, {arg3}] : memref<1x64xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %37[{arg2}, {arg3}] : memref<1x64xf32>""")
#
#
#     print("""%38 = memref.alloca() : memref<1x32xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %8[{arg3}] : memref<32xf32>
# affine.store %{val_id}, %38[{arg2}, {arg3}] : memref<1x32xf32>""")
#
#
#     print("""%39 = memref.alloca() : memref<64x32xf32>""")
#     for arg2 in range(0, 64):
#         for arg3 in range(0, 32):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %7[{arg3}, {arg2}] : memref<32x64xf32>
# affine.store %{val_id}, %39[{arg2}, {arg3}] : memref<64x32xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             for arg4 in range(0, 64):
#                 val_id += 10
#                 print(f"""%{val_id} = affine.load %37[{arg2}, {arg4}] : memref<1x64xf32>
# %{val_id+1} = affine.load %39[{arg4}, {arg3}] : memref<64x32xf32>
# %{val_id+2} = affine.load %38[{arg2}, {arg3}] : memref<1x32xf32>
# %{val_id+3} = arith.mulf %{val_id}, %{val_id+1} : f32
# %{val_id+4} = arith.addf %{val_id+2}, %{val_id+3} : f32
# affine.store %{val_id+4}, %38[{arg2}, {arg3}] : memref<1x32xf32>""")
#
#
#     print("""%40 = memref.alloca() : memref<1x32xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 32):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %38[%c0, {arg3}] : memref<1x32xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %40[{arg2}, {arg3}] : memref<1x32xf32>""")
#
#
#     print("""%41 = memref.alloca() : memref<1x16xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 16):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %10[{arg3}] : memref<16xf32>
# affine.store %{val_id}, %41[{arg2}, {arg3}] : memref<1x16xf32>""")
#
#
#     print("""%42 = memref.alloca() : memref<32x16xf32>""")
#     for arg2 in range(0, 32):
#         for arg3 in range(0, 16):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %9[{arg3}, {arg2}] : memref<16x32xf32>
# affine.store %{val_id}, %42[{arg2}, {arg3}] : memref<32x16xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 16):
#             for arg4 in range(0, 32):
#                 val_id += 10
#                 print(f"""%{val_id} = affine.load %40[{arg2}, {arg4}] : memref<1x32xf32>
# %{val_id+1} = affine.load %42[{arg4}, {arg3}] : memref<32x16xf32>
# %{val_id+2} = affine.load %41[{arg2}, {arg3}] : memref<1x16xf32>
# %{val_id+3} = arith.mulf %{val_id}, %{val_id+1} : f32
# %{val_id+4} = arith.addf %{val_id+2}, %{val_id+3} : f32
# affine.store %{val_id+4}, %41[{arg2}, {arg3}] : memref<1x16xf32>""")
#
#
#     print("""%43 = memref.alloca() : memref<1x16xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 16):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %41[%c0, {arg3}] : memref<1x16xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %43[{arg2}, {arg3}] : memref<1x16xf32>""")
#
#
#     print("""%44 = memref.alloca() : memref<1x8xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %12[{arg3}] : memref<8xf32>
# affine.store %{val_id}, %44[{arg2}, {arg3}] : memref<1x8xf32>""")
#
#
#     print("""%45 = memref.alloca() : memref<16x8xf32>""")
#     for arg2 in range(0, 16):
#         for arg3 in range(0, 8):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %11[{arg3}, {arg2}] : memref<8x16xf32>
# affine.store %{val_id}, %45[{arg2}, {arg3}] : memref<16x8xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             for arg4 in range(0, 16):
#                 val_id += 10
#                 print(f"""%{val_id} = affine.load %43[{arg2}, {arg4}] : memref<1x16xf32>
# %{val_id+1} = affine.load %45[{arg4}, {arg3}] : memref<16x8xf32>
# %{val_id+2} = affine.load %44[{arg2}, {arg3}] : memref<1x8xf32>
# %{val_id+3} = arith.mulf %{val_id}, %{val_id+1} : f32
# %{val_id+4} = arith.addf %{val_id+2}, %{val_id+3} : f32
# affine.store %{val_id+4}, %44[{arg2}, {arg3}] : memref<1x8xf32>""")
#
#
#     print("""%46 = memref.alloca() : memref<1x8xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 8):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %44[%c0, {arg3}] : memref<1x8xf32>
# %{val_id+1} = arith.cmpf ugt, %{val_id}, %cst : f32
# %{val_id+2} = arith.select %{val_id+1}, %{val_id}, %cst : f32
# %{val_id+3} = arith.select %{val_id+1}, %cst, %{val_id} : f32
# %{val_id+4} = arith.truncf %cst_1 : f64 to f32
# %{val_id+5} = arith.mulf %{val_id+3}, %{val_id+4} : f32
# %{val_id+6} = arith.addf %{val_id+2}, %{val_id+5} : f32
# affine.store %{val_id+6}, %46[{arg2}, {arg3}] : memref<1x8xf32>""")
#
#
#     print("""%47 = memref.alloca() : memref<1x2xf32>""")
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 2):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %14[{arg3}] : memref<2xf32>
# affine.store %{val_id}, %arg1[{arg2}, {arg3}] : memref<1x2xf32>""")
#
#
#     print("""%48 = memref.alloca() : memref<8x2xf32>""")
#     for arg2 in range(0, 8):
#         for arg3 in range(0, 2):
#             val_id += 10
#             print(f"""%{val_id} = affine.load %13[{arg3}, {arg2}] : memref<2x8xf32>
# affine.store %{val_id}, %48[{arg2}, {arg3}] : memref<8x2xf32>""")
#
#     for arg2 in range(0, 1):
#         for arg3 in range(0, 2):
#             for arg4 in range(0, 8):
#                 val_id += 10
#                 print(f"""%{val_id} = affine.load %46[{arg2}, {arg4}] : memref<1x8xf32>
# %{val_id+1} = affine.load %48[{arg4}, {arg3}] : memref<8x2xf32>
# %{val_id+2} = affine.load %arg1[{arg2}, {arg3}] : memref<1x2xf32>
# %{val_id+3} = arith.mulf %{val_id}, %{val_id+1} : f32
# %{val_id+4} = arith.addf %{val_id+2}, %{val_id+3} : f32
# affine.store %{val_id+4}, %arg1[{arg2}, {arg3}] : memref<1x2xf32>""")

    print("return\n}\n}")
    
fun()

