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
#include "MemoryPlanning.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <iostream>

using namespace mlir;
using namespace mlir::torch::HLS;

using LiveRanges = DenseMap<Operation *, llvm::SmallVector<size_t>>;

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

template <class Container> int product(const Container &container) {
  auto i = container.begin();
  int prod = 1;

  while (i != container.end())
    prod *= *i++;

  return prod;
}

namespace {

class PromoteAllocsPass : public HLSPromoteAllocsBase<PromoteAllocsPass> {

  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](FuncOp func) {
      LiveRanges liveRanges_ = liveRanges(func);
      SortedUniqueLiveRangeMap<size_t> sortedLiveRanges;
      for (const auto &item : liveRanges_) {
        auto op = llvm::cast<memref::AllocOp>(item.getFirst());
        auto lvr = item.getSecond();
        MemRefType type = op.getType();
        sortedLiveRanges.insert(
            {{{lvr[0], lvr[1]}, op, ""}, product(getConcreteShape(op, func))});
      }
      std::vector<PlannedAlloc> plannedAllocs =
          greedyBySizeWithSmallestGap(sortedLiveRanges);
      hoistAllocs(func, plannedAllocs);
    });
  }

  std::vector<int64_t> getConcreteShape(memref::AllocOp op, FuncOp func) {
    MemRefType type = op.getType();
    std::vector<int64_t> shape;
    auto dynamicSizes = op.getDynamicSizes();
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
      shape.push_back(dimSize);
    }
    return shape;
  }

  LiveRanges liveRanges(FuncOp func) {
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
    DenseMap<Operation *, size_t> opToTopoIndex;
    for (int i = 0; i < sortedOps.size(); ++i) {
      opToTopoIndex[sortedOps[i]] = i;
    }
    LiveRanges allocLiveRange;
    for (const auto &item : opToTopoIndex) {
      auto op = item.first;
      if (!llvm::isa<memref::AllocOp>(op))
        continue;
      for (Operation *user : op->getUsers()) {
        auto idx = opToTopoIndex.find(user)->getSecond();
        allocLiveRange[op].push_back(idx);
      }
    }
    for (auto &item : allocLiveRange) {
      size_t start = *std::min_element(item.second.begin(), item.second.end());
      size_t end = *std::max_element(item.second.begin(), item.second.end());
      llvm::SmallVector<size_t> liveRange{start, end};
      allocLiveRange[item.first] = liveRange;
    }

    return allocLiveRange;
  }

  void hoistAllocs(FuncOp func, std::vector<PlannedAlloc> plannedAllocs) {

    auto inputSlab =
        MemRefType::get({1000000}, IntegerType::get(func.getContext(), 8));
    func.insertArgument(func.getNumArguments(), inputSlab, {});
    BlockArgument slabArg = func.getArgument(func.getNumArguments() - 1);
    OpBuilder builder(func.getBody());

    size_t maxSize = 0;
    for (const auto &pAlloc : plannedAllocs) {
      auto op = llvm::cast<memref::AllocOp>(pAlloc.ulvr.id);
      MemRefType type = op.getType();
      auto shape = getConcreteShape(op, func);
      auto promotedType =
          MemRefType::get(shape, type.getElementType(), type.getLayout(),
                          type.getMemorySpace());

      builder.setInsertionPoint(op);
      auto offsetConstant = builder.create<arith::ConstantIndexOp>(
          op->getLoc(), pAlloc.reg.offset);
      maxSize = std::max(maxSize, pAlloc.reg.nextOffset());
      SmallVector<Value, 4> newOperands;
      builder.setInsertionPoint(op);
      Value view = builder.create<memref::ViewOp>(
          op->getLoc(), MemRefType::get(shape, type.getElementType()), slabArg,
          offsetConstant, newOperands);
      op.replaceAllUsesWith(view);
    }

    auto correctInputSlab =
        MemRefType::get({static_cast<long>(maxSize * 4)},
                        IntegerType::get(func.getContext(), 8));
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
