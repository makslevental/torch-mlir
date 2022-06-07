#ifndef BraggHLS_MLIR_TRANSFORMS_PASSES
#define HLS_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def BraggHLSDropPublicReturn : Pass<"bragg-hls-drop-public-return", "ModuleOp"> {
  let constructor = "mlir::BraggHLS::createBraggHLSDropPublicReturnPass()";
    let description = [{
      vitis for some reason doesn't like return statements? no error description there so just do this instead
    }];
}

def BraggHLSPromoteAllocs : Pass<"bragg-hls-promote-allocs", "ModuleOp"> {
  let constructor = "mlir::BraggHLS::createBraggHLSPromoteAllocsPass()";
    let description = [{
    }];

  let options = [
    Option<"hoistAll", "hoist-all", "bool",/*default=*/"false", "">,
    Option<"hoistGlobals", "hoist-globals", "bool",/*default=*/"false", "">,
    Option<"hoistAffine", "hoist-affine", "bool",/*default=*/"false", "">,
    ];
}

#endif
