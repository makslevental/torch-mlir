#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "PassDetail.h"
#include "Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::HLS;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

static LogicalResult verifyLinalgCompatibleTypes(Operation *op,
                                                 PatternRewriter &rewriter) {
  // Check the value tensor is ranked as expected by Linalg.
  // TODO: Remove this check but use a separate verification pass to verify the
  // invariants expected by later passes.
  auto isValidLinalgType = [](Type type) {
    auto tensor = type.dyn_cast<ValueTensorType>();
    return !tensor ||
           tensor.toBuiltinTensor().dyn_cast_or_null<RankedTensorType>();
  };

  bool valid = llvm::all_of(op->getOperandTypes(), isValidLinalgType) &&
               llvm::all_of(op->getResultTypes(), isValidLinalgType);
  if (!valid)
    return rewriter.notifyMatchFailure(op, "type cannot be lowered to linalg");
  return success();
}

static Value getDimOp(OpBuilder &b, Location loc, Value v, int dim) {
  if (auto tensorType = v.getType().cast<RankedTensorType>()) {
    if (!tensorType.isDynamicDim(dim))
      return b.create<arith::ConstantOp>(
          loc, b.getIndexAttr(tensorType.getShape()[dim]));
  }
  return b.create<tensor::DimOp>(loc, v, dim);
}

static Value castIntToIndex(OpBuilder &b, Location loc, Value v) {
  assert(v.getType().isa<IntegerType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
}

static Value castIndexToInt(OpBuilder &b, Location loc, Value idx) {
  assert(idx.getType().isa<IndexType>() && "must be called with integer type");
  return b.create<arith::IndexCastOp>(loc, b.getI64Type(), idx);
}

static void checkDimEqualHelper(OpBuilder &b, Location loc, Value lhsDim,
                                Value rhsDim) {
  Type lhsType = lhsDim.getType();
  Type rhsType = rhsDim.getType();
  auto checkIntOrIndex = [](Type type) {
    assert(type.isa<IntegerType>() ||
           type.isa<IndexType>() && "must be either integer or index type");
  };
  checkIntOrIndex(lhsType);
  checkIntOrIndex(rhsType);
  Value lhsDimInt = lhsType.isIndex() ? castIndexToInt(b, loc, lhsDim) : lhsDim;
  Value rhsDimInt = rhsType.isIndex() ? castIndexToInt(b, loc, rhsDim) : rhsDim;
  Value contractingDimEqual = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lhsDimInt, rhsDimInt);
  b.create<AssertOp>(loc, contractingDimEqual,
                     b.getStringAttr("mismatching contracting dimension"));
}

namespace {
class ConvertAtenMmOutOp : public OpConversionPattern<AtenMmOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.mat2();
    Value out = adaptor.out();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    if (lhs.getType().cast<RankedTensorType>().getRank() != 2 ||
        rhs.getType().cast<RankedTensorType>().getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.mm to be rank 2");
    }

    Value lhsDim0 = rewriter.create<tensor::DimOp>(loc, lhs, 0);
    Value lhsDim1 = rewriter.create<tensor::DimOp>(loc, lhs, 1);
    Value rhsDim0 = rewriter.create<tensor::DimOp>(loc, rhs, 0);
    Value rhsDim1 = rewriter.create<tensor::DimOp>(loc, rhs, 1);
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, lhsDim1, rhsDim0);
    rewriter.create<AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("out"));
    Value matmul =
        rewriter
            .create<linalg::MatmulOp>(loc, out.getType(), ValueRange{lhs, rhs},
                                      out, variant)
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMatmulOutOp
    : public OpConversionPattern<AtenMatmulOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMatmulOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.other();
    Value out = adaptor.out();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    unsigned lhsRank = lhs.getType().cast<RankedTensorType>().getRank();
    unsigned rhsRank = rhs.getType().cast<RankedTensorType>().getRank();

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();

    // The different cases of torch_matmul op is mentioned here:
    // https://pytorch.org/docs/stable/generated/torch.matmul.html

    // First Case: Dot Product.
    if (lhsRank == 1 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);

      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value dotProd = rewriter
                          .create<linalg::DotOp>(loc, out.getType(),
                                                 ValueRange{lhs, rhs}, out)
                          .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, dotProd);
      return success();
    }

    // Second Case: Vec-Mat Multiplication.
    if (lhsRank == 1 && rhsRank == 2) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
      checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

      Value matmul = rewriter
                         .create<linalg::VecmatOp>(loc, out.getType(),
                                                   ValueRange{lhs, rhs}, out)
                         .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Third Case: Matrix-Vec Multiplication.
    if (lhsRank == 2 && rhsRank == 1) {
      Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      Value matmul = rewriter
                         .create<linalg::MatvecOp>(loc, out.getType(),
                                                   ValueRange{lhs, rhs}, out)
                         .getResult(0);
      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
      return success();
    }

    // Fourth Case: Batch-Matrix Multiplication.
    // TODO: Broadcasting of batch dimension is remaining.
    if (lhsRank >= 3 && rhsRank >= 3 && lhsRank == rhsRank) {

      unsigned batchRank = lhsRank - 2;
      SmallVector<Value, 4> resultShape;

      SmallVector<AffineExpr> lhsExpr;
      SmallVector<AffineExpr> rhsExpr;
      SmallVector<AffineExpr> outExpr;
      SmallVector<StringRef> iteratorTypes;

      // Since broadcasting is a TODO, check whether the lhs and rhs batch
      // dimension match.
      for (unsigned i = 0; i < batchRank; i++) {
        Value lhsBatch = getDimOp(rewriter, loc, lhs, i);
        Value rhsBatch = getDimOp(rewriter, loc, rhs, i);
        resultShape.push_back(lhsBatch);
        lhsExpr.push_back(rewriter.getAffineDimExpr(i));
        rhsExpr.push_back(rewriter.getAffineDimExpr(i));
        outExpr.push_back(rewriter.getAffineDimExpr(i));
        iteratorTypes.push_back(getParallelIteratorTypeName());
        checkDimEqualHelper(rewriter, loc, lhsBatch, rhsBatch);
      }

      Value lhsDim0 = getDimOp(rewriter, loc, lhs, batchRank);
      Value lhsDim1 = getDimOp(rewriter, loc, lhs, batchRank + 1);
      Value rhsDim0 = getDimOp(rewriter, loc, rhs, batchRank);
      Value rhsDim1 = getDimOp(rewriter, loc, rhs, batchRank + 1);
      checkDimEqualHelper(rewriter, loc, lhsDim1, rhsDim0);

      // Push the final matrix dimension.
      resultShape.insert(resultShape.end(), {lhsDim0, rhsDim1});

      lhsExpr.insert(lhsExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 1)});
      rhsExpr.insert(rhsExpr.end(), {rewriter.getAffineDimExpr(batchRank + 1),
                                     rewriter.getAffineDimExpr(batchRank + 2)});
      outExpr.insert(outExpr.end(), {rewriter.getAffineDimExpr(batchRank),
                                     rewriter.getAffineDimExpr(batchRank + 2)});

      auto indexingMaps =
          AffineMap::inferFromExprList({lhsExpr, rhsExpr, outExpr});
      iteratorTypes.insert(iteratorTypes.end(),
                           {"parallel", "reduction", "parallel"});

      Value finalRes =
          rewriter
              .create<linalg::GenericOp>(
                  loc, newResultType, ValueRange{lhs, rhs}, out,
                  /*indexingMaps=*/indexingMaps,
                  /*iteratorTypes=*/iteratorTypes,
                  [&](OpBuilder &b, Location loc, ValueRange args) {
                    Value l = args[0], r = args[1], res = args[2];
                    Value mul = b.create<arith::MulFOp>(loc, l, r);
                    Value add = b.create<arith::AddFOp>(loc, mul, res);
                    b.create<linalg::YieldOp>(loc, add);
                  })
              .getResult(0);

      rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, finalRes);
      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
class HLSConvertTorchToLinalg
    : public HLSConvertTorchToLinalgBase<HLSConvertTorchToLinalg> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect, mlir::torch::Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<AtenMmOutOp>();
    patterns.add<ConvertAtenMmOutOp>(typeConverter, context);
    target.addIllegalOp<AtenMatmulOutOp>();
    patterns.add<ConvertAtenMatmulOutOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::HLS::createHLSConvertTorchToLinalgPass() {
  return std::make_unique<HLSConvertTorchToLinalg>();
}
