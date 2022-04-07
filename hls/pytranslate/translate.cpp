//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "EmitHLSPy.h"

int main(int argc, char **argv) {
  mlir::scalehls::registerEmitHLSPyTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "ScaleHLS Translation Tool"));
}
