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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::torch::HLS;



using namespace mlir;

template <typename AnyMemRefDefOp>
static bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
                                    Region *region) {
  auto memRefType = memrefDefOp.getType();
  // Statically shaped.
  if (!memRefType.isDynamicDim(index))
    return true;
  // Get the position of the dimension among dynamic dimensions;
  unsigned dynamicDimPos = memRefType.getDynamicDimIndex(index);
  return isValidSymbol(*(memrefDefOp.getDynamicSizes().begin() + dynamicDimPos),
                       region);
}

namespace {

class PromoteAllocsPass : public HLSPromoteAllocsBase<PromoteAllocsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](FuncOp func) {
      auto sortedOps = topoSort(func);
      hoistAllocs(func, sortedOps);
    });
  }

  SetVector<Operation *> topoSort(FuncOp func) {
    SetVector<Operation *> sortedOps;
    func.walk([&sortedOps, &func](Operation *op) {
      if (op->getName() == func->getName())
        return WalkResult::advance();
      if (op->getParentOp()->getName() != func->getName())
        return WalkResult::skip();

      sortedOps.insert(op);
      return WalkResult::advance();
    });
    sortedOps = topologicalSort(sortedOps);
    // not sure why but above is reverse toposort
    SetVector<Operation *> reversedSortedOps(sortedOps.rbegin(), sortedOps.rend());
    return reversedSortedOps;
  }

  void hoistAllocs(FuncOp func, SetVector<Operation *> sortedOps) {
    auto inputSlab =
        MemRefType::get({1000000}, IntegerType::get(func.getContext(), 8));
    func.insertArgument(func.getNumArguments(), inputSlab, {});
    BlockArgument slabArg = func.getArgument(func.getNumArguments() - 1);
    OpBuilder builder(func.getBody());

//    auto builder = OpBuilder::atBlockBegin(&func.getBody().front());
//    auto zero = builder.createOrFold<arith::ConstantIndexOp>(func.getBody().getLoc(), 0L);
    auto maxSize = 0;
    auto offset = 0;

    for (auto &_op : sortedOps) {
      if (!llvm::isa<memref::AllocOp>(_op))
        continue;

      auto op = llvm::cast<memref::AllocOp>(_op);
      MemRefType type = op.getType();
      std::vector<int64_t> shape;
      auto dynamicSizes = op.getDynamicSizes();

      auto size = 1;
      for (int idx = 0; idx < type.getShape().size(); ++idx) {
        auto dimSize = type.getShape()[idx];
        if (dimSize < 0) {
          if (isMemRefSizeValidSymbol(op, idx, func->getParentRegion())) {
            unsigned dynamicDimPos = type.getDynamicDimIndex(idx);
            dimSize = dynamicSizes[dynamicDimPos]
                .getDefiningOp<arith::ConstantIndexOp>()
                .value();
          } else {
            shape.push_back(-1);
            continue;
          }
        }
        size *= dimSize;
        shape.push_back(dimSize);
      }
      maxSize = std::max(maxSize, size);
      auto promotedType =
          MemRefType::get(shape, type.getElementType(), type.getLayout(),
                          type.getMemorySpace());

      builder.setInsertionPoint(op);
      auto offsetConstant = builder.create<arith::ConstantIndexOp>(op->getLoc(), offset);
      offset += size * 4;
      SmallVector<Value, 4> newOperands;
      builder.setInsertionPoint(op);
      Value view = builder.create<memref::ViewOp>(op->getLoc(),
                                                  MemRefType::get(shape, type.getElementType()), slabArg, offsetConstant,
                                                  newOperands);
      op.replaceAllUsesWith(view);
    }

    auto correctInputSlab =
        MemRefType::get({offset}, IntegerType::get(func.getContext(), 8));
    func.insertArgument(func.getNumArguments(), correctInputSlab, {});
    BlockArgument correctSlabArg = func.getArgument(func.getNumArguments() - 1);
    slabArg.replaceAllUsesWith(correctSlabArg);
    func.eraseArgument(func.getNumArguments() - 2);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::HLS::createHLSPromoteAllocsPass() {
  return std::make_unique<PromoteAllocsPass>();
}
