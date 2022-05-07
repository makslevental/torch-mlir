//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "HLSPassDetail.h"
#include "HLSPasses.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::HLS;

// Helper funtion to get rank of `Base tensor type`.
// -1 is returned if the tensorRank can't be determined.
static int getTensorRank(Value tensor) {
  int tensorRank = -1;
  BaseTensorType tensorType = tensor.getType().cast<BaseTensorType>();

  if (tensorType.hasSizes()) {
    ArrayRef<int64_t> tensorShape = tensorType.getSizes();
    tensorRank = tensorShape.size();
  }
  return tensorRank;
}

namespace {
class DecomposeComplexOpsPass
    : public HLSDecomposeComplexOpsBase<DecomposeComplexOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::HLS::createHLSDecomposeComplexOpsPass() {
  return std::make_unique<DecomposeComplexOpsPass>();
}
