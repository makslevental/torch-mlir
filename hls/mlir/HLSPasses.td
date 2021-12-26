#ifndef HLS_MLIR_TRANSFORMS_PASSES
#define HLS_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def HLSDropPublicReturn : Pass<"torch-hls-drop-public-return", "ModuleOp"> {
  let constructor = "mlir::torch::HLS::createHLSDropPublicReturnPass()";
}

def HLSLinalgBufferize : Pass<"torch-hls-linalg-bufferize", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSLinalgBufferizePass()";
}

def HLSDecomposeComplexOps : Pass<"torch-hls-decompose-complex-ops", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSDecomposeComplexOpsPass()";
}

def HLSRefineTypes : Pass<"torch-hls-refine-types", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSRefineTypesPass()";
}

def HLSConvertTorchToLinalg : Pass<"torch-hls-convert-torch-to-linalg", "FuncOp"> {
  let constructor = "mlir::torch::HLS::createHLSConvertTorchToLinalgPass()";
}

//def HLSConvertOperators : Pass<"torch-hls-convert-operators", "FuncOp"> {
//  let constructor = "mlir::torch::HLS::createHLSConvertOperatorsPass()";
//}


#endif
