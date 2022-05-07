//===- DirtyPass.cpp - Test Linalg codegen strategy -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the Linalg codegen strategy.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include <tuple>
#include <utility>

#include "HLSPassDetail.h"
#include "HLSPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MathExtras.h"
#include <chrono>
#include <ctime>
#include <iostream>

#include "mlir/Dialect/DLTI/DLTI.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct DirtyPassLLVM
    : public PassWrapper<DirtyPassLLVM, OperationPass<mlir::LLVM::LLVMFuncOp>> {
  StringRef getArgument() const final { return "dirty-pass-llvm"; }
  StringRef getDescription() const final { return "Dirty Pass."; }
  DirtyPassLLVM() = default;
  DirtyPassLLVM(const DirtyPassLLVM &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
        mlir::LLVM::LLVMDialect,
        gpu::GPUDialect,
        linalg::LinalgDialect,
        memref::MemRefDialect,
        arith::ArithmeticDialect,
        scf::SCFDialect,
        vector::VectorDialect>();
    // clang-format on
  }

  Option<bool> unrollparfor{*this, "unrollparfor",
                            llvm::cl::desc("unroll the par fors"),
                            llvm::cl::init(true)};

  void runOnOperation() override;
};

struct DirtyPass : public PassWrapper<DirtyPass, OperationPass<func::FuncOp>> {
  StringRef getArgument() const final { return "dirty-pass"; }
  StringRef getDescription() const final { return "Dirty Pass."; }
  DirtyPass() = default;
  DirtyPass(const DirtyPass &pass) : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    mlir::LLVM::LLVMDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    cf::ControlFlowDialect,
                    arith::ArithmeticDialect,
                    scf::SCFDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  Option<bool> unrollparfor{*this, "unrollparfor",
                            llvm::cl::desc("unroll the par fors"),
                            llvm::cl::init(true)};

  Option<bool> csts{*this, "csts", llvm::cl::desc("csts promotion"),
                    llvm::cl::init(true)};
  Option<bool> unrollloops{*this, "unrollloops", llvm::cl::desc("unroll loops"),
                    llvm::cl::init(true)};

  void runOnOperation() override;
};

uint64_t getConstantTripCount(scf::ForOp forOp) {
  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  int64_t lbCst = lbCstOp.value();
  int64_t ubCst = ubCstOp.value();
  int64_t stepCst = stepCstOp.value();
  assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
         "expected positive loop bounds and step");
  int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);
  return tripCount;
}

void replaceCst(llvm::SmallDenseMap<int64_t, Value> constants,
                arith::ConstantIndexOp cst) {
  int64_t cst_val = cst.value();
  auto already_exist = constants[cst_val];
  cst.replaceAllUsesWith(already_exist);
  cst->erase();
}

/// Generates unrolled copies of scf::ForOp 'loopBodyBlock', with
/// associated 'forOpIV' by 'unrollFactor', calling 'ivRemapFn' to remap
/// 'forOpIV' for each unrolled body. If specified, annotates the Ops in each
/// unrolled iteration using annotateFn.
static void generateUnrolledLoop(
    Block *loopBodyBlock, Value forOpIV, uint64_t unrollFactor,
    //    function_ref<Value(unsigned, Value, OpBuilder)> ivRemapFn,
    llvm::SmallDenseMap<int64_t, Value> constants,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
    ValueRange iterArgs, ValueRange yieldedValues) {
  // Builder to insert unrolled bodies just before the terminator of the body of
  // 'forOp'.
  auto builder = OpBuilder::atBlockTerminator(loopBodyBlock);

  if (!annotateFn)
    annotateFn = [](unsigned, Operation *, OpBuilder) {};

  // Keep a pointer to the last non-terminator operation in the original block
  // so that we know what to clone (since we are doing this in-place).
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

  // Unroll the contents of 'forOp' (append unrollFactor - 1 additional copies).
  SmallVector<Value, 4> lastYielded(yieldedValues);

  // TODO(max): this should start at 0???
  for (unsigned i = 1; i < unrollFactor; i++) {
    BlockAndValueMapping operandMap;

    // Prepare operand map.
    operandMap.map(iterArgs, lastYielded);

    // If the induction variable is used, create a remapping to the value for
    // this unrolled instance.
    if (!forOpIV.use_empty()) {
      //      Value ivUnroll = ivRemapFn(i, forOpIV, builder);
      //      operandMap.map(forOpIV, ivUnroll);
      operandMap.map(forOpIV, constants[i]);
    }

    // Clone the original body of 'forOp'.
    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      Operation *clonedOp = builder.clone(*it, operandMap);
      annotateFn(i, clonedOp, builder);
    }

    // Update yielded values.
    for (unsigned i = 0, e = lastYielded.size(); i < e; i++)
      lastYielded[i] = operandMap.lookup(yieldedValues[i]);
  }

  // Make sure we annotate the Ops in the original body. We do this last so that
  // any annotations are not copied into the cloned Ops above.
  for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
    annotateFn(0, &*it, builder);
  }

  // Update operands of the yield statement.
  loopBodyBlock->getTerminator()->setOperands(lastYielded);
}

/// Unrolls 'forOp' by 'unrollFactor', returns success if the loop is unrolled.
LogicalResult loopUnrollByFactor(
    scf::ForOp forOp, uint64_t unrollFactor,
    llvm::SmallDenseMap<int64_t, Value> constants,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  assert(unrollFactor > 0 && "expected positive unroll factor");

  // Return if the loop body is empty.
  if (llvm::hasSingleElement(forOp.getBody()->getOperations()))
    return success();

  // Compute tripCount = ceilDiv((upperBound - lowerBound), step) and populate
  // 'upperBoundUnrolled' and 'stepUnrolled' for static and dynamic cases.
  OpBuilder boundsBuilder(forOp);
  auto loc = forOp.getLoc();
  auto step = forOp.getStep();
  Value upperBoundUnrolled;
  Value stepUnrolled;
  bool generateEpilogueLoop = true;

  auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (lbCstOp && ubCstOp && stepCstOp) {
    // Constant loop bounds computation.
    int64_t lbCst = lbCstOp.value();
    int64_t ubCst = ubCstOp.value();
    int64_t stepCst = stepCstOp.value();
    assert(lbCst >= 0 && ubCst >= 0 && stepCst >= 0 &&
           "expected positive loop bounds and step");
    int64_t tripCount = mlir::ceilDiv(ubCst - lbCst, stepCst);

    if (unrollFactor == 1) {
      if (tripCount == 1 && failed(promoteIfSingleIteration(forOp)))
        return failure();
      return success();
    }

    int64_t tripCountEvenMultiple = tripCount - (tripCount % unrollFactor);
    int64_t upperBoundUnrolledCst = lbCst + tripCountEvenMultiple * stepCst;
    assert(upperBoundUnrolledCst <= ubCst);
    int64_t stepUnrolledCst = stepCst * unrollFactor;

    // Create constant for 'upperBoundUnrolled' and set epilogue loop flag.
    generateEpilogueLoop = upperBoundUnrolledCst < ubCst;
    if (generateEpilogueLoop) {
      upperBoundUnrolled = constants[upperBoundUnrolledCst];
    } else
      upperBoundUnrolled = ubCstOp;

    stepUnrolled =
        stepCst == stepUnrolledCst ? step : constants[stepUnrolledCst];
  } else {
    std::cerr << "exit wtfbbq";
    exit(-1);
  }

  // Create epilogue clean up loop starting at 'upperBoundUnrolled'.
  if (generateEpilogueLoop) {
    std::cerr << "epilogue loop\n";
    OpBuilder epilogueBuilder(forOp->getContext());
    epilogueBuilder.setInsertionPoint(forOp->getBlock(),
                                      std::next(Block::iterator(forOp)));
    auto epilogueForOp = cast<scf::ForOp>(epilogueBuilder.clone(*forOp));
    epilogueForOp.setLowerBound(upperBoundUnrolled);

    // Update uses of loop results.
    auto results = forOp.getResults();
    auto epilogueResults = epilogueForOp.getResults();
    auto epilogueIterOperands = epilogueForOp.getIterOperands();

    for (auto e : llvm::zip(results, epilogueResults, epilogueIterOperands)) {
      std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
      epilogueForOp->replaceUsesOfWith(std::get<2>(e), std::get<0>(e));
    }
    (void)promoteIfSingleIteration(epilogueForOp);
  }

  // Create unrolled loop.
  forOp.setUpperBound(upperBoundUnrolled);
  forOp.setStep(stepUnrolled);

  auto iterArgs = ValueRange(forOp.getRegionIterArgs());
  for (const auto &item : iterArgs) {
    std::cerr << "iterargs\n";
    item.getDefiningOp()->dump();
  }
  auto yieldedValues = forOp.getBody()->getTerminator()->getOperands();
  for (const auto &item : yieldedValues) {
    std::cerr << "yieldedValues\n";
    item.getDefiningOp()->dump();
  }

  auto start = std::chrono::system_clock::now();
  // Some computation here
  ::generateUnrolledLoop(
      forOp.getBody(), forOp.getInductionVar(), unrollFactor,
      //      [&](unsigned i, Value iv, OpBuilder b) {
      //        // iv' = iv + step * i;
      ////        auto ivval = lbCstOp.value();
      ////        return constants[(int64_t)(ivval +
      /// step.getDefiningOp<arith::ConstantIndexOp>().value() * i)];
      //
      ////        auto stride = b.create<arith::MulIOp>(
      ////            loc, step, b.create<arith::ConstantIndexOp>(loc, i));
      //        auto c = b.create<arith::AddIOp>(loc, iv,
      //        constants[(int64_t)(step.getDefiningOp<arith::ConstantIndexOp>().value()
      //        * i)]); return c;
      //      },
      constants, annotateFn, iterArgs, yieldedValues);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "unroll loop elapsed time: " << (end - start).count() << "s\n";

  // Promote the loop body up if this has turned into a single iteration loop.
  std::vector<Operation *> opsToErase;
  start = std::chrono::system_clock::now();
  forOp.walk([&](arith::AddIOp addOp) {
    if (addOp.getLhs().getDefiningOp<arith::ConstantIndexOp>() &&
        addOp.getRhs().getDefiningOp<arith::ConstantIndexOp>()) {
      addOp.getResult().replaceAllUsesWith(
          constants[(int64_t)(addOp.getLhs()
                                  .getDefiningOp<arith::ConstantIndexOp>()
                                  .value() +
                              addOp.getRhs()
                                  .getDefiningOp<arith::ConstantIndexOp>()
                                  .value())]);

      if (isOpTriviallyDead(addOp)) {
        opsToErase.push_back(addOp);
      }
    }
  });
  for (auto *op : opsToErase)
    op->erase();
  end = std::chrono::system_clock::now();
  std::cout << "dce addp elapsed time: " << (end - start).count() << "s\n";

  start = std::chrono::system_clock::now();
  (void)promoteIfSingleIteration(forOp);
  end = std::chrono::system_clock::now();
  std::cout << "promoteifsingle  elapsed time: " << (end - start).count()
            << "s\n";
  return success();
}

} // namespace

std::vector<std::vector<int64_t>>
cart_product(const std::vector<std::vector<int64_t>> &v) {
  std::vector<std::vector<int64_t>> s = {{}};
  for (const auto &u : v) {
    std::vector<std::vector<int64_t>> r;
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = move(r);
  }
  return s;
}

void unrollParFor(scf::ParallelOp parForOp,
                  llvm::SmallDenseMap<int64_t, Value> constants) {
  auto ivs = parForOp.getInductionVars();
  auto lbs = parForOp.getLowerBound();
  auto ubs = parForOp.getUpperBound();

  auto parentBlock = parForOp->getBlock();
  auto loopBodyBlock = parForOp.getBody();
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);
  auto builder = OpBuilder::atBlockTerminator(parentBlock);

  std::vector<std::vector<int64_t>> inds(ubs.size());
  for (int i = 0; i < ubs.size(); ++i) {
    auto ub = ubs[i].getDefiningOp<arith::ConstantIndexOp>().value();
    for (int j = 0; j < ub; ++j) {
      inds[i].push_back(j);
    }
  }
  inds = cart_product(inds);
  Operation *clonedOp;
  for (const auto &item : inds) {
    BlockAndValueMapping operandMap;

    for (int i = 0; i < item.size(); ++i) {
      auto val = item[i];
      auto iv = ivs[i];
      operandMap.map(iv, constants[val]);
    }

    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      clonedOp = builder.clone(*it, operandMap);
      if (auto memloadop = llvm::dyn_cast<memref::LoadOp>(clonedOp)) {
        for (const auto &idx : memloadop.getIndices()) {
          if (auto applyopd = idx.getDefiningOp<AffineApplyOp>()) {
            auto lhs = applyopd.getOperand(0)
                           .getDefiningOp<arith::ConstantIndexOp>()
                           .value();
            auto rhs = applyopd.getOperand(1)
                           .getDefiningOp<arith::ConstantIndexOp>()
                           .value();
            applyopd.replaceAllUsesWith(constants[(int64_t)(lhs + rhs)]);
            applyopd->erase();
          }
        }
      }
    }
  }
  loopBodyBlock->getTerminator()->erase();
  parForOp.erase();
}


void unrollParForLLVM(scf::ParallelOp parForOp,
                  llvm::SmallDenseMap<int64_t, Value> constants) {
  auto ivs = parForOp.getInductionVars();
  auto lbs = parForOp.getLowerBound();
  auto ubs = parForOp.getUpperBound();

  auto parentBlock = parForOp->getBlock();
  auto loopBodyBlock = parForOp.getBody();
  Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);
  auto builder = OpBuilder::atBlockTerminator(parentBlock);

  std::vector<std::vector<int64_t>> inds(ubs.size());
  for (int i = 0; i < ubs.size(); ++i) {
    // all ivs are not unrealized conversion casts
    auto ub = ubs[i].getDefiningOp()->getOpOperand(0).get()
        .getDefiningOp<LLVM::ConstantOp>()
        .getValueAttr()
        .cast<IntegerAttr>()
        .getInt();
    for (int j = 0; j < ub; ++j) {
      inds[i].push_back(j);
    }
  }
  inds = cart_product(inds);
  Operation *clonedOp;
  for (const auto &item : inds) {
    BlockAndValueMapping operandMap;

    for (int i = 0; i < item.size(); ++i) {
      auto val = item[i];
      auto iv = ivs[i];
      operandMap.map(iv, constants[val]);
    }

    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      clonedOp = builder.clone(*it, operandMap);
      for (auto item : clonedOp->getOperands()) {
        if (auto applyopd = item.getDefiningOp<AffineApplyOp>()) {
          applyopd->dump();
          auto lhs = applyopd.getOperand(0)
              .getDefiningOp<arith::ConstantIndexOp>()
              .value();
          auto rhs = applyopd.getOperand(1)
              .getDefiningOp<arith::ConstantIndexOp>()
              .value();
          applyopd.replaceAllUsesWith(constants[(int64_t)(lhs + rhs)]);
          applyopd->erase();
        }

      }
    }
  }
  loopBodyBlock->getTerminator()->erase();
  parForOp.erase();
}


/// Apply transformations specified as patterns.
void DirtyPass::runOnOperation() {
  auto funcOp = getOperation();
  auto ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&funcOp.getBody().getBlocks().front());

  std::vector<arith::ConstantIndexOp> csts;
  funcOp.walk<WalkOrder::PreOrder>(
      [&](arith::ConstantIndexOp cst) { csts.emplace_back(cst); });
  llvm::SmallDenseMap<int64_t, Value> constants;

  if (this->csts) {
    for (int i = 0; i < 3000; ++i) {
      auto val = builder.create<arith::ConstantIndexOp>(funcOp->getLoc(), i)
                     .getResult();
      constants.insert(std::make_pair(i, val));
    }
  }

  for (int i = 1; i < csts.size(); ++i) {
    replaceCst(constants, csts[i]);
  }

  std::vector<memref::AllocaOp> allocs;
  int64_t maxnumel = 0;
  funcOp.walk<WalkOrder::PreOrder>(
      [&](memref::AllocaOp cst) { allocs.emplace_back(cst); });
  for (int i = 0; i < allocs.size(); ++i) {
    allocs[i]->moveBefore(&*funcOp.getBody().getOps().begin());
    maxnumel = std::max(maxnumel, allocs[i].getType().getNumElements());
  }

  if (unrollloops) {
    funcOp.walk<WalkOrder::PostOrder>([&](scf::ForOp forOp) {
      ::loopUnrollByFactor(forOp, getConstantTripCount(forOp), constants,
                           nullptr);
    });
  }

  if (unrollparfor) {
    funcOp.walk<WalkOrder::PreOrder>(
        [&](scf::ParallelOp forOp) { unrollParFor(forOp, constants); });
  }

  funcOp.walk([&](cf::AssertOp op) { op->erase(); });
}

void eraseAddsMul(LLVM::LLVMFuncOp funcOp,
                  llvm::SmallDenseMap<int64_t, Value> llvm_csts) {
  std::vector<Operation*> toErase;
  for (Operation &item : funcOp.getBody().getOps()) {
    if (auto addop = llvm::dyn_cast<LLVM::AddOp>(item)) {
      if (addop->getName().stripDialect().contains("add")) {
        if (addop.getLhs().getDefiningOp<LLVM::ConstantOp>() &&
            addop.getRhs().getDefiningOp<LLVM::ConstantOp>() &&
            addop.getLhs().getType().isa<IntegerType>()) {
          auto lhs = addop.getLhs()
                         .getDefiningOp<LLVM::ConstantOp>()
                         .getValueAttr()
                         .cast<IntegerAttr>()
                         .getInt();
          auto rhs = addop.getRhs()
                         .getDefiningOp<LLVM::ConstantOp>()
                         .getValueAttr()
                         .cast<IntegerAttr>()
                         .getInt();
          llvm_csts[lhs + rhs].dump();
          addop->getResult(0).replaceAllUsesWith(llvm_csts[lhs + rhs]);
//          auto it = addop->getIterator();
//          for (int i = 0; i < 1000; ++i) {
//            it++;
//            for (int j = 0; j < it->getNumOperands(); ++j) {
//              auto& oper = it->getOpOperand(j);
//              if (oper.get() == res) {
//                it->setOperand(j, llvm_csts[lhs+rhs]);
//              }
//            }
//          }
          toErase.emplace_back(addop->getResult(0).getDefiningOp());
        }
      }
    }
    if (auto addop = llvm::dyn_cast<LLVM::MulOp>(item)) {
      if (addop->getName().stripDialect().contains("mul")) {
        if (addop.getLhs().getDefiningOp<LLVM::ConstantOp>() &&
            addop.getRhs().getDefiningOp<LLVM::ConstantOp>() &&
            addop.getLhs().getType().isa<IntegerType>()) {
          auto lhs = addop.getLhs()
                         .getDefiningOp<LLVM::ConstantOp>()
                         .getValueAttr()
                         .cast<IntegerAttr>()
                         .getInt();
          auto rhs = addop.getRhs()
                         .getDefiningOp<LLVM::ConstantOp>()
                         .getValueAttr()
                         .cast<IntegerAttr>()
                         .getInt();
          llvm_csts[lhs * rhs].dump();
          addop->getResult(0).replaceAllUsesWith(llvm_csts[lhs * rhs]);
//          auto it = addop->getIterator();
//          for (int i = 0; i < 1000; ++i) {
//            it++;
//            for (int j = 0; j < it->getNumOperands(); ++j) {
//              auto& oper = it->getOpOperand(j);
//              if (oper.get() == res) {
//                it->setOperand(j, llvm_csts[lhs*rhs]);
//              }
//            }
//          }
          toErase.emplace_back(addop->getResult(0).getDefiningOp());
        }
      }
    }
  }

  for (auto &item : toErase)
    item->erase();
}

void DirtyPassLLVM::runOnOperation() {
  auto funcOp = getOperation();
  llvm::SmallDenseMap<int64_t, Value> llvm_csts;
  funcOp.walk<WalkOrder::PreOrder>([&](LLVM::ConstantOp cstOp) {
    if (auto oAttr = cstOp.getValueAttr().dyn_cast<IntegerAttr>()) {
      cstOp->dump();
      auto v = oAttr.getInt();
      if (!llvm_csts.count(v)) {
        llvm_csts.insert(std::make_pair(v, cstOp.getResult()));
      }
      cstOp->moveBefore(&*funcOp.getBody().getOps().begin());
    }
  });

  if (unrollparfor) {
    funcOp.walk<WalkOrder::PreOrder>(
        [&](scf::ParallelOp forOp) { unrollParForLLVM(forOp, llvm_csts); });
  }

  std::vector<std::pair<Value, Value>> toReplace;
  funcOp.walk<WalkOrder::PreOrder>([&](LLVM::ConstantOp cstOp) {
    if (auto oAttr = cstOp.getValueAttr().dyn_cast<IntegerAttr>()) {
      cstOp->dump();
      auto v = oAttr.getInt();
      if (llvm_csts.count(v) && llvm_csts[v] != cstOp) {
        toReplace.emplace_back(cstOp, llvm_csts[v]);
      }
    }
  });
  for (auto &item : toReplace) {
    item.first.replaceAllUsesWith(item.second);
    item.first.getDefiningOp<LLVM::ConstantOp>()->erase();
  }

//  eraseAddsMul(funcOp, llvm_csts);
}

namespace mlir {
namespace torch {
namespace HLS {
void registerDirtyPass() {
  PassRegistration<DirtyPass>();
  PassRegistration<DirtyPassLLVM>();
}
} // namespace HLS
} // namespace torch
} // namespace mlir
