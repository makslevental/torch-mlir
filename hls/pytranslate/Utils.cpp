//===----------------------------------------------------------------------===//
//
// Copyright 2020-2021 The ScaleHLS Authors.
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace scalehls;

//===----------------------------------------------------------------------===//
// Memory and loop analysis utils
//===----------------------------------------------------------------------===//

/// Parse array attributes.
SmallVector<int64_t, 8> scalehls::getIntArrayAttrValue(Operation *op,
                                                       StringRef name) {
  SmallVector<int64_t, 8> array;
  if (auto arrayAttr = op->getAttrOfType<ArrayAttr>(name)) {
    for (auto attr : arrayAttr)
      if (auto intAttr = attr.dyn_cast<IntegerAttr>())
        array.push_back(intAttr.getInt());
      else
        return SmallVector<int64_t, 8>();
    return array;
  } else
    return SmallVector<int64_t, 8>();
}

/// Collect all load and store operations in the block and return them in "map".
void scalehls::getMemAccessesMap(Block &block, MemAccessesMap &map,
                                 bool includeVectorTransfer) {
  for (auto &op : block) {
    if (auto load = dyn_cast<AffineReadOpInterface>(op))
      map[load.getMemRef()].push_back(&op);

    else if (auto store = dyn_cast<AffineWriteOpInterface>(op))
      map[store.getMemRef()].push_back(&op);

    else if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      if (includeVectorTransfer)
        map[read.getSource()].push_back(&op);

    } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      if (includeVectorTransfer)
        map[write.getSource()].push_back(&op);

    } else if (op.getNumRegions()) {
      // Recursively collect memory access operations in each block.
      for (auto &region : op.getRegions())
        for (auto &block : region)
          getMemAccessesMap(block, map);
    }
  }
}

// Check if the lhsOp and rhsOp are in the same block. If so, return their
// ancestors that are located at the same block. Note that in this check,
// AffineIfOp is transparent.
Optional<std::pair<Operation *, Operation *>>
scalehls::checkSameLevel(Operation *lhsOp, Operation *rhsOp) {
  // If lhsOp and rhsOp are already at the same level, return true.
  if (lhsOp->getBlock() == rhsOp->getBlock())
    return std::pair<Operation *, Operation *>(lhsOp, rhsOp);

  // Helper to get all surrounding AffineIfOps.
  auto getSurroundIfs =
      ([&](Operation *op, SmallVector<Operation *, 4> &nests) {
        nests.push_back(op);
        auto currentOp = op;
        while (true) {
          auto parentOp = currentOp->getParentOp();
          if (isa<AffineIfOp, scf::IfOp>(parentOp)) {
            nests.push_back(parentOp);
            currentOp = parentOp;
          } else
            break;
        }
      });

  SmallVector<Operation *, 4> lhsNests;
  SmallVector<Operation *, 4> rhsNests;

  getSurroundIfs(lhsOp, lhsNests);
  getSurroundIfs(rhsOp, rhsNests);

  // If any parent of lhsOp and any parent of rhsOp are at the same level,
  // return true.
  for (auto lhs : lhsNests)
    for (auto rhs : rhsNests)
      if (lhs->getBlock() == rhs->getBlock())
        return std::pair<Operation *, Operation *>(lhs, rhs);

  return Optional<std::pair<Operation *, Operation *>>();
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned scalehls::getCommonSurroundingLoops(Operation *A, Operation *B,
                                             AffineLoopBand *band) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getLoopIVs(*A, &loopsA);
  getLoopIVs(*B, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
    if (band != nullptr)
      band->push_back(loopsB[i]);
  }
  return numCommonLoops;
}

/// Calculate the lower and upper bound of the affine map if possible.
Optional<std::pair<int64_t, int64_t>>
scalehls::getBoundOfAffineMap(AffineMap map, ValueRange operands) {
  if (map.isSingleConstant()) {
    auto constBound = map.getSingleConstantResult();
    return std::pair<int64_t, int64_t>(constBound, constBound);
  }

  // For now, we can only handle one result value map.
  if (map.getNumResults() != 1)
    return Optional<std::pair<int64_t, int64_t>>();

  auto context = map.getContext();
  SmallVector<int64_t, 4> lbs;
  SmallVector<int64_t, 4> ubs;
  for (auto operand : operands) {
    // Only if the affine map operands are induction variable, the calculation
    // is possible.
    if (!isForInductionVar(operand))
      return Optional<std::pair<int64_t, int64_t>>();

    // Only if the owner for op of the induction variable has constant bound,
    // the calculation is possible.
    auto forOp = getForInductionVarOwner(operand);
    if (!forOp.hasConstantBounds())
      return Optional<std::pair<int64_t, int64_t>>();

    auto lb = forOp.getConstantLowerBound();
    auto ub = forOp.getConstantUpperBound();
    auto step = forOp.getStep();

    lbs.push_back(lb);
    ubs.push_back(ub - 1 - (ub - 1 - lb) % step);
  }

  // TODO: maybe a more efficient algorithm.
  auto operandNum = operands.size();
  SmallVector<int64_t, 16> results;
  for (unsigned i = 0, e = pow(2, operandNum); i < e; ++i) {
    SmallVector<AffineExpr, 4> replacements;
    for (unsigned pos = 0; pos < operandNum; ++pos) {
      if (i >> pos % 2 == 0)
        replacements.push_back(getAffineConstantExpr(lbs[pos], context));
      else
        replacements.push_back(getAffineConstantExpr(ubs[pos], context));
    }
    auto newExpr = map.getResult(0).replaceDimsAndSymbols(replacements, {});

    if (auto constExpr = newExpr.dyn_cast<AffineConstantExpr>())
      results.push_back(constExpr.getValue());
    else
      return Optional<std::pair<int64_t, int64_t>>();
  }

  auto minmax = std::minmax_element(results.begin(), results.end());
  return std::pair<int64_t, int64_t>(*minmax.first, *minmax.second);
}

bool scalehls::isFullyPartitioned(MemRefType memrefType) {
  if (memrefType.getRank() == 0)
    return true;

  bool fullyPartitioned = false;
  SmallVector<int64_t, 8> factors;
  getPartitionFactors(memrefType, &factors);

  auto shapes = memrefType.getShape();
  fullyPartitioned =
      factors == SmallVector<int64_t, 8>(shapes.begin(), shapes.end());

  return fullyPartitioned;
}

// Calculate partition factors through analyzing the "memrefType" and return
// them in "factors". Meanwhile, the overall partition number is calculated and
// returned as well.
int64_t scalehls::getPartitionFactors(MemRefType memrefType,
                                      SmallVector<int64_t, 8> *factors) {
  auto shape = memrefType.getShape();
  auto layoutMap = memrefType.getLayout().getAffineMap();
  int64_t accumFactor = 1;

  for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
    int64_t factor = 1;
    auto expr = layoutMap.getResult(dim);

    if (auto binaryExpr = expr.dyn_cast<AffineBinaryOpExpr>())
      if (auto rhsExpr = binaryExpr.getRHS().dyn_cast<AffineConstantExpr>()) {
        if (expr.getKind() == AffineExprKind::Mod)
          factor = rhsExpr.getValue();
        else if (expr.getKind() == AffineExprKind::FloorDiv)
          factor = (shape[dim] + rhsExpr.getValue() - 1) / rhsExpr.getValue();
      }

    accumFactor *= factor;
    if (factors != nullptr)
      factors->push_back(factor);
  }

  return accumFactor;
}

/// This is method for finding the number of child loops which immediatedly
/// contained by the input operation.
unsigned scalehls::getChildLoopNum(Operation *op) {
  unsigned childNum = 0;
  for (auto &region : op->getRegions())
    for (auto &block : region)
      for (auto &op : block)
        if (isa<AffineForOp>(op))
          ++childNum;

  return childNum;
}

/// Get the whole loop band given the innermost loop and return it in "band".
static void getLoopBandFromInnermost(AffineForOp forOp, AffineLoopBand &band) {
  band.clear();
  AffineLoopBand reverseBand;

  auto currentLoop = forOp;
  while (true) {
    reverseBand.push_back(currentLoop);

    auto parentLoop = currentLoop->getParentOfType<AffineForOp>();
    if (!parentLoop)
      break;

    if (getChildLoopNum(parentLoop) == 1)
      currentLoop = parentLoop;
    else
      break;
  }

  band.append(reverseBand.rbegin(), reverseBand.rend());
}

/// Given a tiled loop band, return true and get the tile (tile-space) loop band
/// and the point (intra-tile) loop band. If failed, return false.
bool scalehls::getTileAndPointLoopBand(const AffineLoopBand &band,
                                       AffineLoopBand &tileBand,
                                       AffineLoopBand &pointBand) {
  tileBand.clear();
  pointBand.clear();
  bool isPointLoop = false;

  for (auto loop : band) {
    if (!isPointLoop)
      tileBand.push_back(loop);
    else if (isPointLoop)
      pointBand.push_back(loop);
    else if (!isPointLoop) {
      isPointLoop = true;
      pointBand.push_back(loop);

    } else {
      tileBand.clear();
      pointBand.clear();
      return false;
    }
  }
  return true;
}

/// Get the whole loop band given the outermost loop and return it in "band".
/// Meanwhile, the return value is the innermost loop of this loop band.
AffineForOp scalehls::getLoopBandFromOutermost(AffineForOp forOp,
                                               AffineLoopBand &band) {
  band.clear();
  auto currentLoop = forOp;
  while (true) {
    band.push_back(currentLoop);

    if (getChildLoopNum(currentLoop) == 1)
      currentLoop = *currentLoop.getOps<AffineForOp>().begin();
    else
      break;
  }
  return band.back();
}

/// Collect all loop bands in the "block" and return them in "bands". If
/// "allowHavingChilds" is true, loop bands containing more than 1 other loop
/// bands are also collected. Otherwise, only loop bands that contains no child
/// loops are collected.
void scalehls::getLoopBands(Block &block, AffineLoopBands &bands,
                            bool allowHavingChilds) {
  bands.clear();
  block.walk([&](AffineForOp loop) {
    auto childNum = getChildLoopNum(loop);

    if (childNum == 0 || (childNum > 1 && allowHavingChilds)) {
      AffineLoopBand band;
      getLoopBandFromInnermost(loop, band);
      bands.push_back(band);
    }
  });
}

void scalehls::getArrays(Block &block, SmallVectorImpl<Value> &arrays,
                         bool allowArguments) {
  // Collect argument arrays.
  if (allowArguments)
    for (auto arg : block.getArguments()) {
      if (arg.getType().isa<MemRefType>())
        arrays.push_back(arg);
    }

  // Collect local arrays.
  for (auto &op : block.getOperations()) {
    if (isa<memref::AllocaOp, memref::AllocOp>(op))
      arrays.push_back(op.getResult(0));
  }
}

Optional<unsigned> scalehls::getAverageTripCount(AffineForOp forOp) {
  if (auto optionalTripCount = getConstantTripCount(forOp))
    return optionalTripCount.getValue();
  else {
    // TODO: A temporary approach to estimate the trip count. For now, we take
    // the average of the upper bound and lower bound of trip count as the
    // estimated trip count.
    auto lowerBound = getBoundOfAffineMap(forOp.getLowerBoundMap(),
                                          forOp.getLowerBoundOperands());
    auto upperBound = getBoundOfAffineMap(forOp.getUpperBoundMap(),
                                          forOp.getUpperBoundOperands());

    if (lowerBound && upperBound) {
      auto lowerTripCount =
          upperBound.getValue().second - lowerBound.getValue().first;
      auto upperTripCount =
          upperBound.getValue().first - lowerBound.getValue().second;
      return (lowerTripCount + upperTripCount + 1) / 2;
    } else
      return Optional<unsigned>();
  }
}

bool scalehls::checkDependence(Operation *A, Operation *B) {
  AffineLoopBand commonLoops;
  unsigned numCommonLoops = getCommonSurroundingLoops(A, B, &commonLoops);

  // Traverse each loop level to find dependencies.
  for (unsigned depth = numCommonLoops; depth > 0; depth--) {
    // Skip all parallel loop level.
    FlatAffineValueConstraints depConstrs;
    DependenceResult result = checkMemrefAccessDependence(
        MemRefAccess(A), MemRefAccess(B), depth, &depConstrs,
        /*dependenceComponents=*/nullptr);
    if (hasDependence(result))
      return true;
  }

  return false;
}

/// Localize each tosa/arith constant to right before its each use. Only
/// localize the constants whose size is below the bitsThreshold.
void scalehls::localizeConstants(Block &block, int64_t bitsThreshold) {
  auto builder = OpBuilder(block.getParentOp());

  // Collect all constants.
  SmallVector<Operation *, 16> constants;
  block.walk([&](Operation *constant) {
    if (isa<tosa::ConstOp, arith::ConstantOp>(constant)) {
      auto type = constant->getResult(0).getType();
      if (auto shapedType = type.dyn_cast<ShapedType>()) {
        if (shapedType.getSizeInBits() <= bitsThreshold)
          constants.push_back(constant);
      } else
        constants.push_back(constant);
    }
  });

  // Localize constants to each of its use.
  for (auto constant : constants) {
    for (auto &use : llvm::make_early_inc_range(constant->getUses())) {
      builder.setInsertionPoint(use.getOwner());
      auto cloneConstant = builder.clone(*constant);
      use.set(cloneConstant->getResult(0));
    }
    constant->erase();
  }
}

FuncOp scalehls::getTopFunc(ModuleOp module, std::string topFuncName) {
  FuncOp topFunc;
  for (auto func : module.getOps<FuncOp>())
    if (func.getName() == topFuncName) {
      if (!topFunc)
        topFunc = func;
      else
        return FuncOp();
    }
  return topFunc;
}

FuncOp scalehls::getRuntimeFunc(ModuleOp module,
                                      std::string runtimeFuncName) {
  FuncOp runtimeFunc;
  for (auto func : module.getOps<FuncOp>())
    if (func.getName() == runtimeFuncName) {
      if (!runtimeFunc)
        runtimeFunc = func;
      else
        return FuncOp();
    }
  return runtimeFunc;
}

//===----------------------------------------------------------------------===//
// PtrLikeMemRefAccess Struct Definition
//===----------------------------------------------------------------------===//

PtrLikeMemRefAccess::PtrLikeMemRefAccess(Operation *loadOrStoreOpInst) {
  Operation *opInst = nullptr;
  SmallVector<Value, 4> indices;

  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();

    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();

    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }

  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);

  accessMap.reset(map, operands);
}

bool PtrLikeMemRefAccess::operator==(const PtrLikeMemRefAccess &rhs) const {
  if (memref != rhs.memref || impl != rhs.impl)
    return false;

  if (impl == rhs.impl && impl && rhs.impl)
    return true;

  AffineValueMap diff;
  AffineValueMap::difference(accessMap, rhs.accessMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}
