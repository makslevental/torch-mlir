//===----------------------------------------------------------------------===//
//
// Forked/modified from https://github.com/hanchenye/scalehls/
//
//===----------------------------------------------------------------------===//

#include "EmitBraggHLSPy.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::bragghls::registerEmitHLSPyTranslation();

  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "BraggHLS Translation Tool"));
}
