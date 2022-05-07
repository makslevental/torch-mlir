#ifndef HLS_MLIR_TRANSFORMS_PASSES
#define HLS_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def HLSDropPublicReturn : Pass<"torch-hls-drop-public-return", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSDropPublicReturnPass()";
    let description = [{
      vitis for some reason doesn't like return statements? no error description there so just do this instead
    }];
}

def HLSLinalgBufferize : Pass<"torch-hls-linalg-bufferize", "func::FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSLinalgBufferizePass()";
  let description = [{
    don't allocate new buffers for ops labeled as a "out variant"
  }];
}

def HLSDecomposeComplexOps : Pass<"torch-hls-decompose-complex-ops", "func::FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSDecomposeComplexOpsPass()";
    let description = [{
      decompose matmulout
    }];
}

def HLSRefineTypes : Pass<"torch-hls-refine-types", "func::FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSRefineTypesPass()";
  let description = [{
    for some reason trying to handle refines (for new ops) in sequence doesn't work
    so have to do all of them at once, including new ones
  }];
}

def HLSConvertTorchToLinalg : Pass<"convert-torch-hls-to-linalg", "func::FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSConvertTorchToLinalgPass()";
  let description = [{
    handle new ops
  }];
}

def HLSPromoteAllocs : Pass<"torch-hls-promote-allocs", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSPromoteAllocsPass()";
    let description = [{
    }];

  let options = [
    Option<"hoistAll", "hoist-all", "bool",/*default=*/"false", "">,
    Option<"hoistGlobals", "hoist-globals", "bool",/*default=*/"false", "">,
    Option<"hoistAffine", "hoist-affine", "bool",/*default=*/"false", "">,
    ];
}

def HLSHLSQuantizeModule : Pass<"torch-hls-quantize-module", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSHLSQuantizeModulePass()";
  let description = [{
  }];
}

def HLSQuantize : Pass<"torch-hls-quantize", "func::FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSQuantizePass()";
  let description = [{
  }];
}

def ConvertCopyToAffineLoops : Pass<"torch-hls-convert-copy-to-affine-loops", "func::FuncOp"> {
  let summary = "Convert copy and assign to affine loops";
  let description = [{
      This convert-copy-to-affine-loops pass will first remove redundant copies
                                              and lower assigns to copies, and then convert copies to affine loops.
  }];

  let constructor = "mlir::torch::HLS::createConvertCopyToAffineLoopsPass()";
}

//def SimplifyMemrefAccess : Pass<"simplify-memref-access", "func::FuncOp"> {
//  let summary = "Remove redundant load and store operations";
//  let description = [{
//    This simplify-memref-access pass will eliminate the known redundant load and
//    store operations for simplifying the memref access.
//  }];
//
//  let constructor = "mlir::torch::HLS::createSimplifyMemrefAccessPass()";
//}

#endif
