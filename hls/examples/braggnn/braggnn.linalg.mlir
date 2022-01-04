#loc0 = loc(unknown)
#map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
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
  func @forward(%arg0: memref<1x1x11x11xf32> loc(unknown)) -> memref<?x?xf32> {
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc2)
    %cst_0 = arith.constant 1.000000e-02 : f64 loc(#loc3)
    %c1 = arith.constant 1 : index loc(#loc4)
    %c7 = arith.constant 7 : index loc(#loc4)
    %true = arith.constant true loc(#loc5)
    %c5 = arith.constant 5 : index loc(#loc2)
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
    assert %true, "expect groups to be 1" loc(#loc5)
    %15 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc5)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : memref<64xf32>) outs(%15 : memref<1x64x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc5)
    } loc(#loc5)
    %16 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc5)
    linalg.copy(%15, %16) : memref<1x64x9x9xf32>, memref<1x64x9x9xf32>  loc(#loc5)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %0 : memref<1x1x11x11xf32>, memref<64x1x3x3xf32>) outs(%16 : memref<1x64x9x9xf32>) loc(#loc5)
    assert %true, "expect groups to be 1" loc(#loc6)
    %17 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc6)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : memref<32xf32>) outs(%17 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc6)
    } loc(#loc6)
    %18 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc6)
    linalg.copy(%17, %18) : memref<1x32x9x9xf32>, memref<1x32x9x9xf32>  loc(#loc6)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%16, %1 : memref<1x64x9x9xf32>, memref<32x64x1x1xf32>) outs(%18 : memref<1x32x9x9xf32>) loc(#loc6)
    assert %true, "expect groups to be 1" loc(#loc7)
    %19 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc7)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : memref<32xf32>) outs(%19 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc7)
    } loc(#loc7)
    %20 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc7)
    linalg.copy(%19, %20) : memref<1x32x9x9xf32>, memref<1x32x9x9xf32>  loc(#loc7)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%16, %1 : memref<1x64x9x9xf32>, memref<32x64x1x1xf32>) outs(%20 : memref<1x32x9x9xf32>) loc(#loc7)
    assert %true, "expect groups to be 1" loc(#loc8)
    %21 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc8)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : memref<32xf32>) outs(%21 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc8)
    } loc(#loc8)
    %22 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc8)
    linalg.copy(%21, %22) : memref<1x32x9x9xf32>, memref<1x32x9x9xf32>  loc(#loc8)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%16, %1 : memref<1x64x9x9xf32>, memref<32x64x1x1xf32>) outs(%22 : memref<1x32x9x9xf32>) loc(#loc8)
    %23 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc9)
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %20 : memref<1x32x9x9xf32>, memref<1x32x9x9xf32>) outs(%23 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown), %arg3: f32 loc(unknown)):  // no predecessors
      %62 = arith.mulf %arg1, %arg2 : f32 loc(#loc9)
      linalg.yield %62 : f32 loc(#loc9)
    } loc(#loc9)
    %24 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc10)
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : memref<1x32x9x9xf32>) outs(%24 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = math.exp %arg1 : f32 loc(#loc10)
      linalg.yield %62 : f32 loc(#loc10)
    } loc(#loc10)
    %25 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc11)
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23 : memref<1x32x9x9xf32>) outs(%25 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = math.exp %arg1 : f32 loc(#loc11)
      linalg.yield %62 : f32 loc(#loc11)
    } loc(#loc11)
    %26 = memref.alloc() : memref<1x32x1x1xf32> loc(#loc11)
    linalg.fill(%cst, %26) : f32, memref<1x32x1x1xf32>  loc(#loc11)
    %27 = memref.alloc() : memref<1x32x1x1xf32> loc(#loc11)
    linalg.copy(%26, %27) : memref<1x32x1x1xf32>, memref<1x32x1x1xf32>  loc(#loc11)
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%25 : memref<1x32x9x9xf32>) outs(%27 : memref<1x32x1x1xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.addf %arg1, %arg2 : f32 loc(#loc11)
      linalg.yield %62 : f32 loc(#loc11)
    } loc(#loc11)
    %28 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc10)
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %27 : memref<1x32x9x9xf32>, memref<1x32x1x1xf32>) outs(%28 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown), %arg3: f32 loc(unknown)):  // no predecessors
      %62 = arith.divf %arg1, %arg2 : f32 loc(#loc10)
      linalg.yield %62 : f32 loc(#loc10)
    } loc(#loc10)
    %29 = memref.alloc() : memref<1x32x9x9xf32> loc(#loc12)
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %22 : memref<1x32x9x9xf32>, memref<1x32x9x9xf32>) outs(%29 : memref<1x32x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown), %arg3: f32 loc(unknown)):  // no predecessors
      %62 = arith.mulf %arg1, %arg2 : f32 loc(#loc12)
      linalg.yield %62 : f32 loc(#loc12)
    } loc(#loc12)
    assert %true, "expect groups to be 1" loc(#loc13)
    %30 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc13)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : memref<64xf32>) outs(%30 : memref<1x64x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc13)
    } loc(#loc13)
    %31 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc13)
    linalg.copy(%30, %31) : memref<1x64x9x9xf32>, memref<1x64x9x9xf32>  loc(#loc13)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%29, %2 : memref<1x32x9x9xf32>, memref<64x32x1x1xf32>) outs(%31 : memref<1x64x9x9xf32>) loc(#loc13)
    %32 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc14)
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %16 : memref<1x64x9x9xf32>, memref<1x64x9x9xf32>) outs(%32 : memref<1x64x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown), %arg3: f32 loc(unknown)):  // no predecessors
      %62 = arith.sitofp %c1_i64 : i64 to f32 loc(#loc14)
      %63 = arith.mulf %arg2, %62 : f32 loc(#loc14)
      %64 = arith.addf %arg1, %63 : f32 loc(#loc14)
      linalg.yield %64 : f32 loc(#loc14)
    } loc(#loc14)
    %33 = memref.alloc() : memref<1x64x9x9xf32> loc(#loc4)
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : memref<1x64x9x9xf32>) outs(%33 : memref<1x64x9x9xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc4)
      %63 = select %62, %arg1, %cst : f32 loc(#loc4)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc4)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc4)
      %66 = arith.mulf %64, %65 : f32 loc(#loc4)
      %67 = arith.addf %63, %66 : f32 loc(#loc4)
      linalg.yield %67 : f32 loc(#loc4)
    } loc(#loc4)
    assert %true, "expect groups to be 1" loc(#loc2)
    %34 = memref.alloc() : memref<1x32x7x7xf32> loc(#loc2)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8 : memref<32xf32>) outs(%34 : memref<1x32x7x7xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc2)
    } loc(#loc2)
    %35 = memref.alloc() : memref<1x32x7x7xf32> loc(#loc2)
    linalg.copy(%34, %35) : memref<1x32x7x7xf32>, memref<1x32x7x7xf32>  loc(#loc2)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%33, %3 : memref<1x64x9x9xf32>, memref<32x64x3x3xf32>) outs(%35 : memref<1x32x7x7xf32>) loc(#loc2)
    %36 = memref.alloc(%c1, %c7, %c7) : memref<?x32x?x?xf32> loc(#loc4)
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : memref<1x32x7x7xf32>) outs(%36 : memref<?x32x?x?xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc4)
      %63 = select %62, %arg1, %cst : f32 loc(#loc4)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc4)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc4)
      %66 = arith.mulf %64, %65 : f32 loc(#loc4)
      %67 = arith.addf %63, %66 : f32 loc(#loc4)
      linalg.yield %67 : f32 loc(#loc4)
    } loc(#loc4)
    assert %true, "expect groups to be 1" loc(#loc2)
    %37 = memref.alloc(%c1, %c5, %c5) : memref<?x8x?x?xf32> loc(#loc2)
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12 : memref<8xf32>) outs(%37 : memref<?x8x?x?xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc2)
    } loc(#loc2)
    %38 = memref.alloc(%c1, %c5, %c5) : memref<?x8x?x?xf32> loc(#loc2)
    linalg.copy(%37, %38) : memref<?x8x?x?xf32>, memref<?x8x?x?xf32>  loc(#loc2)
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%36, %4 : memref<?x32x?x?xf32>, memref<8x32x3x3xf32>) outs(%38 : memref<?x8x?x?xf32>) loc(#loc2)
    %39 = memref.alloc(%c1, %c5, %c5) : memref<?x8x?x?xf32> loc(#loc4)
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : memref<?x8x?x?xf32>) outs(%39 : memref<?x8x?x?xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc4)
      %63 = select %62, %arg1, %cst : f32 loc(#loc4)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc4)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc4)
      %66 = arith.mulf %64, %65 : f32 loc(#loc4)
      %67 = arith.addf %63, %66 : f32 loc(#loc4)
      linalg.yield %67 : f32 loc(#loc4)
    } loc(#loc4)
    %40 = memref.cast %39 : memref<?x8x?x?xf32> to memref<?x?x?x?xf32> loc(#loc4)
    %41 = memref.collapse_shape %40 [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32> loc(#loc15)
    assert %true, "mismatching contracting dimension for aten.linear" loc(#loc16)
    %42 = memref.alloc(%c1) : memref<?x64xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%6 : memref<64xf32>) outs(%42 : memref<?x64xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %43 = memref.alloc() : memref<200x64xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%5 : memref<64x200xf32>) outs(%43 : memref<200x64xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %44 = memref.alloc(%c1) : memref<?x64xf32> loc(#loc16)
    linalg.copy(%42, %44) : memref<?x64xf32>, memref<?x64xf32>  loc(#loc16)
    linalg.matmul ins(%41, %43 : memref<?x?xf32>, memref<200x64xf32>) outs(%44 : memref<?x64xf32>) loc(#loc16)
    %45 = memref.alloc(%c1) : memref<?x64xf32> loc(#loc1)
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%44 : memref<?x64xf32>) outs(%45 : memref<?x64xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc1)
      %63 = select %62, %arg1, %cst : f32 loc(#loc1)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc1)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
      %66 = arith.mulf %64, %65 : f32 loc(#loc1)
      %67 = arith.addf %63, %66 : f32 loc(#loc1)
      linalg.yield %67 : f32 loc(#loc1)
    } loc(#loc1)
    %46 = memref.alloc(%c1) : memref<?x32xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%8 : memref<32xf32>) outs(%46 : memref<?x32xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %47 = memref.alloc() : memref<64x32xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%7 : memref<32x64xf32>) outs(%47 : memref<64x32xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %48 = memref.alloc(%c1) : memref<?x32xf32> loc(#loc16)
    linalg.copy(%46, %48) : memref<?x32xf32>, memref<?x32xf32>  loc(#loc16)
    linalg.matmul ins(%45, %47 : memref<?x64xf32>, memref<64x32xf32>) outs(%48 : memref<?x32xf32>) loc(#loc16)
    %49 = memref.alloc(%c1) : memref<?x32xf32> loc(#loc1)
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%48 : memref<?x32xf32>) outs(%49 : memref<?x32xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc1)
      %63 = select %62, %arg1, %cst : f32 loc(#loc1)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc1)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
      %66 = arith.mulf %64, %65 : f32 loc(#loc1)
      %67 = arith.addf %63, %66 : f32 loc(#loc1)
      linalg.yield %67 : f32 loc(#loc1)
    } loc(#loc1)
    %50 = memref.alloc(%c1) : memref<?x16xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%10 : memref<16xf32>) outs(%50 : memref<?x16xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %51 = memref.alloc() : memref<32x16xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%9 : memref<16x32xf32>) outs(%51 : memref<32x16xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %52 = memref.alloc(%c1) : memref<?x16xf32> loc(#loc16)
    linalg.copy(%50, %52) : memref<?x16xf32>, memref<?x16xf32>  loc(#loc16)
    linalg.matmul ins(%49, %51 : memref<?x32xf32>, memref<32x16xf32>) outs(%52 : memref<?x16xf32>) loc(#loc16)
    %53 = memref.alloc(%c1) : memref<?x16xf32> loc(#loc1)
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%52 : memref<?x16xf32>) outs(%53 : memref<?x16xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc1)
      %63 = select %62, %arg1, %cst : f32 loc(#loc1)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc1)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
      %66 = arith.mulf %64, %65 : f32 loc(#loc1)
      %67 = arith.addf %63, %66 : f32 loc(#loc1)
      linalg.yield %67 : f32 loc(#loc1)
    } loc(#loc1)
    %54 = memref.alloc(%c1) : memref<?x8xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%12 : memref<8xf32>) outs(%54 : memref<?x8xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %55 = memref.alloc() : memref<16x8xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%11 : memref<8x16xf32>) outs(%55 : memref<16x8xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %56 = memref.alloc(%c1) : memref<?x8xf32> loc(#loc16)
    linalg.copy(%54, %56) : memref<?x8xf32>, memref<?x8xf32>  loc(#loc16)
    linalg.matmul ins(%53, %55 : memref<?x16xf32>, memref<16x8xf32>) outs(%56 : memref<?x8xf32>) loc(#loc16)
    %57 = memref.alloc(%c1) : memref<?x8xf32> loc(#loc1)
    linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%56 : memref<?x8xf32>) outs(%57 : memref<?x8xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      %62 = arith.cmpf ugt, %arg1, %cst : f32 loc(#loc1)
      %63 = select %62, %arg1, %cst : f32 loc(#loc1)
      %64 = select %62, %cst, %arg1 : f32 loc(#loc1)
      %65 = arith.truncf %cst_0 : f64 to f32 loc(#loc1)
      %66 = arith.mulf %64, %65 : f32 loc(#loc1)
      %67 = arith.addf %63, %66 : f32 loc(#loc1)
      linalg.yield %67 : f32 loc(#loc1)
    } loc(#loc1)
    %58 = memref.alloc(%c1) : memref<?x2xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%14 : memref<2xf32>) outs(%58 : memref<?x2xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %59 = memref.alloc() : memref<8x2xf32> loc(#loc16)
    linalg.generic {indexing_maps = [#map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%13 : memref<2x8xf32>) outs(%59 : memref<8x2xf32>) {
    ^bb0(%arg1: f32 loc(unknown), %arg2: f32 loc(unknown)):  // no predecessors
      linalg.yield %arg1 : f32 loc(#loc16)
    } loc(#loc16)
    %60 = memref.alloc(%c1) : memref<?x2xf32> loc(#loc16)
    linalg.copy(%58, %60) : memref<?x2xf32>, memref<?x2xf32>  loc(#loc16)
    linalg.matmul ins(%57, %59 : memref<?x8xf32>, memref<8x2xf32>) outs(%60 : memref<?x2xf32>) loc(#loc16)
    %61 = memref.cast %60 : memref<?x2xf32> to memref<?x?xf32> loc(#loc16)
    return %61 : memref<?x?xf32> loc(#loc0)
  } loc(#loc0)
} loc(#loc0)
#loc1 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1578:17 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":105:15))
#loc2 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc3 = loc(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:35 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc4 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1578:17 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/activation.py":748:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":103:15))
#loc5 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":101:15))
#loc6 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":35:16) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc7 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":36:14) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc8 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":37:12) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc9 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":39:20 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc10 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":40:20 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc11 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":40:38 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc12 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":41:22 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc13 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":443:15 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/conv.py":447:15) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":43:19) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc14 = loc(callsite("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":44:19 at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":102:15))
#loc15 = loc("/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":104:15)
#loc16 = loc(callsite(callsite(callsite("/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/functional.py":1951:11 at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/linear.py":103:15) at "/home/mlevental/dev_projects/torch-mlir/mlir_venv/lib64/python3.9/site-packages/torch/nn/modules/container.py":141:20) at "/home/mlevental/dev_projects/torch-mlir/hls/python/braggnn.py":105:15))
