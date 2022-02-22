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

def HLSLinalgBufferize : Pass<"torch-hls-linalg-bufferize", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSLinalgBufferizePass()";
  let description = [{
    don't allocate new buffers for ops labeled as a "out variant"
  }];
}

def HLSDecomposeComplexOps : Pass<"torch-hls-decompose-complex-ops", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSDecomposeComplexOpsPass()";
    let description = [{
      decompose matmulout
    }];
}

def HLSRefineTypes : Pass<"torch-hls-refine-types", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSRefineTypesPass()";
  let description = [{
    for some reason trying to handle refines (for new ops) in sequence doesn't work
    so have to do all of them at once, including new ones
  }];
}

def HLSConvertTorchToLinalg : Pass<"convert-torch-hls-to-linalg", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSConvertTorchToLinalgPass()";
  let description = [{
    handle new ops
  }];
}

def HLSPromoteAllocs : Pass<"torch-hls-promote-allocs", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSPromoteAllocsPass()";
    let description = [{
    }];
}

def HLSHLSQuantizeModule : Pass<"torch-hls-quantize-module", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSHLSQuantizeModulePass()";
  let description = [{
  }];
}

def HLSQuantize : Pass<"torch-hls-quantize", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSQuantizePass()";
  let description = [{
  }];
}

#endif
