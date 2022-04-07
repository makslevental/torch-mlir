//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#ifndef SCALEHLS_TRANSLATION_EMITHLSPy_H
#define SCALEHLS_TRANSLATION_EMITHLSPy_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace scalehls {

LogicalResult emitHLSPy(ModuleOp module, llvm::raw_ostream &os);
void registerEmitHLSPyTranslation();

} // namespace scalehls
} // namespace mlir

#endif // SCALEHLS_TRANSLATION_EMITHLSPy_H
