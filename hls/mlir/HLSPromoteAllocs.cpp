//===- RefinePublicReturn.cpp ------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "HLSPassDetail.h"
#include "HLSPasses.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::torch::HLS;

namespace {

class PromoteAllocsPass : public HLSPromoteAllocsBase<PromoteAllocsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](FuncOp func) {
      rewriteSignature(func);
    });
  }

  void rewriteSignature(FuncOp func) {
    // Find the unique return op.
    ReturnOp returnOp;
    WalkResult walkResult = func.walk([&](memref::AllocOp op) {
      func.insertArgument(func.getNumArguments(), op.getType(), {});
      BlockArgument arg = func.getArgument(func.getNumArguments()-1);
      op.replaceAllUsesWith(arg);
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      func.emitError() << "unimplemented: refining returns for function with "
                          "more than one return op";
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::HLS::createHLSPromoteAllocsPass() {
  return std::make_unique<PromoteAllocsPass>();
}
