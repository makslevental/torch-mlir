#include "HLSPassDetail.h"
#include "HLSPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::linalg;
using namespace mlir::torch::HLS;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

template <typename T>
Value createGetTensorLiteral(OpBuilder &b, Location loc,
                             ArrayRef<int64_t> shape, ArrayRef<T> vals,
                             Type elementType) {
  auto type_ = RankedTensorType::get(shape, elementType);
  //  auto initAttr = DenseElementsAttr::get(type_,
  //  {1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f});
  auto valsAttr = DenseElementsAttr::get(type_, vals);
  Value tensor = b.getContext()
                     ->getLoadedDialect<tensor::TensorDialect>()
                     ->materializeConstant(b, valsAttr, type_, loc)
                     ->getResult(0);
  return tensor;
}

template <typename T>
Value createGetTensorLiteral(OpBuilder &b, Location loc,
                             ArrayRef<int64_t> shape, T val, Type elementType) {
  int64_t numElts = 1;
  for (const auto &item : shape)
    numElts *= item;
  std::vector<T> vals_(numElts, val);
  ArrayRef<T> vals(vals_);
  //  ArrayRef<T> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  return createGetTensorLiteral(b, loc, shape, vals, elementType);
}

static LogicalResult checkNotNone(PatternRewriter &rewriter, Operation *op,
                                  Value v) {
  Type type = v.getType();
  if (type.isa<OptionalType>() || type.isa<Torch::NoneType>() ||
      type.isa<mlir::NoneType>())
    return rewriter.notifyMatchFailure(op, "unimplemented None type arg");
  return success();
}

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

static SmallVector<Value>
getAsConstantIntValues(OpBuilder &b, Location loc,
                       SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(ints, [&](int64_t val) -> Value {
    return b.create<arith::ConstantOp>(loc,
                                       b.getIntegerAttr(b.getI64Type(), val));
  }));
}

static SmallVector<OpFoldResult>
getAsOpFoldResult(OpBuilder &b, Location loc, SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(llvm::map_range(
      ints, [&](int64_t val) -> OpFoldResult { return b.getIndexAttr(val); }));
}

static Value getPaddedTensor(Operation *op, OpBuilder &b, Value &input,
                             SmallVectorImpl<int64_t> &paddingInts) {
  assert(input.getType().isa<RankedTensorType>() &&
         "input must be RankedTensorType");
  Location loc = op->getLoc();
  Value c0 = b.create<arith::ConstantOp>(
      loc,
      b.getZeroAttr(input.getType().cast<RankedTensorType>().getElementType()));
  SmallVector<OpFoldResult> paddings = getAsOpFoldResult(b, loc, paddingInts);
  Type ranked4DTensorType = linalg::PadTensorOp::inferResultType(
      input.getType().cast<RankedTensorType>(), paddingInts, paddingInts);
  Value paddedInput = linalg::PadTensorOp::createPadScalarOp(
      ranked4DTensorType, input, c0, /*low=*/paddings, /*high=*/paddings,
      /*packing=*/false, loc, b);
  return paddedInput;
}

static bool isConstantIntListMatching(Value value,
                                      SmallVectorImpl<int64_t> &expects) {
  SmallVector<int64_t> intValues;
  if (!matchPattern(value, m_TorchConstantIntList(intValues)))
    return false;

  if (intValues.size() != expects.size())
    return false;

  for (auto it : llvm::zip(intValues, expects)) {
    if (std::get<0>(it) != std::get<1>(it))
      return false;
  }
  return true;
}

// Normalization formula:
//   ((input - mean) / sqrt(var + eps)) * weight + bias
static Value createLinalgPayloadCalculationForNormOps(
    OpBuilder &b, Location loc, Type elemTy, Value input, Value mean, Value var,
    Value eps, Value weight, Value bias) {
  Value inputSubMean = b.create<arith::SubFOp>(loc, input, mean);
  // The eps is always f64.
  Value truncatedEps = b.create<arith::TruncFOp>(loc, elemTy, eps);
  Value varPlusEps = b.create<arith::AddFOp>(loc, var, truncatedEps);
  Value rSTD = b.create<math::RsqrtOp>(loc, varPlusEps);
  Value temp = b.create<arith::MulFOp>(loc, inputSubMean, rSTD);
  Value timesWeight = b.create<arith::MulFOp>(loc, temp, weight);
  Value plusBias = b.create<arith::AddFOp>(loc, timesWeight, bias);
  return plusBias;
}

// Helper function to caculate the output tensor dims for convolution-like ops.
// Along each dim:
// dim_out =
//  floor((dim_in + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1
static Value getOutputDimForConvOps(OpBuilder &b, Location loc, Value in,
                                    Value paddingInt, Value dilationInt,
                                    Value kernelSizeInt, Value strideInt) {
  Value c1 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(1));
  Value c2 = b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(2));

  Value doublePadding = b.create<arith::MulIOp>(loc, paddingInt, c2);
  // in + 2 * padding
  Value inAddDoublePadding =
      b.create<arith::AddIOp>(loc, castIndexToInt(b, loc, in), doublePadding);

  // dilation * (kernelSize - 1)
  Value kernelSizeSub1 = b.create<arith::SubIOp>(loc, kernelSizeInt, c1);
  Value dilationTimesKernelSize =
      b.create<arith::MulIOp>(loc, dilationInt, kernelSizeSub1);

  Value temp =
      b.create<arith::SubIOp>(loc, inAddDoublePadding, dilationTimesKernelSize);
  Value dividend = b.create<arith::SubIOp>(loc, temp, c1);
  Value division = b.create<arith::FloorDivSIOp>(loc, dividend, strideInt);
  Value out = b.create<arith::AddIOp>(loc, division, c1);
  return castIntToIndex(b, loc, out);
}

Value createLinalgPayloadCalculationForElementwiseOp(OpBuilder &b, Location loc,
                                                     TypeConverter *converter,
                                                     ValueRange payloadArgs,
                                                     Operation *op,
                                                     ArrayRef<Value> operands) {
  if (auto relu = dyn_cast<AtenReluOp>(op)) {
    if (!relu.getType()
             .cast<ValueTensorType>()
             .getDtype()
             .isa<mlir::FloatType>()) {
      relu.emitError("unimplemented: non-floating point dtype");
      return nullptr;
    }
    Type elementType = payloadArgs[0].getType();
    Value constZero =
        b.create<arith::ConstantOp>(loc, b.getZeroAttr(elementType));
    Value pred = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UGT,
                                         payloadArgs[0], constZero);
    return b.create<SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForElementwiseOp");
  return nullptr;
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
class ConvertAtenMatmulOutOp : public OpConversionPattern<AtenMatmulOutOp> {
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
class ConvertAtenTHNNConv2dOutOp
    : public OpConversionPattern<AtenTHNNConv2dOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenTHNNConv2dOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value out = adaptor.out();       /* in form of F*C*H*W */

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, intType, v);
    };

    Value N = getDimOp(rewriter, loc, input, 0);
    Value Hin = getDimOp(rewriter, loc, input, 2);
    Value Win = getDimOp(rewriter, loc, input, 3);
    Value F = getDimOp(rewriter, loc, weight, 0);
    Value weightH = getDimOp(rewriter, loc, weight, 2);
    Value weightW = getDimOp(rewriter, loc, weight, 3);

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t> paddingInts;
    if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }

    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t, 2> dilationInts = {1, 1};
    // Pad the input tensor according to padding.
    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput =
        getPaddedTensor(op, rewriter, input, paddingIncludingNC);

    SmallVector<Value> paddingIntValues =
        getAsConstantIntValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    Value Hout = getOutputDimForConvOps(
        rewriter, loc, Hin, paddingIntValues[0], dilationIntValues[0],
        castIndexToInt(weightH), strideIntValues[0]);
    Value Wout = getOutputDimForConvOps(
        rewriter, loc, Win, paddingIntValues[1], dilationIntValues[1],
        castIndexToInt(weightW), strideIntValues[1]);

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{N, F, Hout, Wout}, elementType);

    Value bias = adaptor.bias();
    Value biasInitTensor;
    if (bias.getType().isa<Torch::NoneType>()) {
      Value c0float = rewriter.create<arith::ConstantOp>(
          loc, FloatAttr::get(elementType, 0.0));
      biasInitTensor = rewriter.create<linalg::FillOp>(loc, c0float, initTensor)
                           .getResult(0);
    } else {
      auto biasType = bias.getType().cast<RankedTensorType>();
      if (biasType.getRank() != 1)
        return rewriter.notifyMatchFailure(op, "expect bias to be rank 1");
      if (elementType != biasType.getElementType())
        return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");

      auto resultRank = initTensor.getType().cast<RankedTensorType>().getRank();
      SmallVector<AffineMap> indexingMaps = {
          // bias is used to initialize the channels - dimension 1 of output
          AffineMap::get(/*dimCount=*/resultRank, /*symbolCount=*/0,
                         rewriter.getAffineDimExpr(1), context),
          rewriter.getMultiDimIdentityMap(resultRank)};
      SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
      biasInitTensor = rewriter
                           .create<linalg::GenericOp>(
                               loc, initTensor.getType(), bias, initTensor,
                               indexingMaps, iteratorTypes,
                               [](OpBuilder &b, Location loc, ValueRange args) {
                                 b.create<linalg::YieldOp>(loc, args[0]);
                               })
                           .getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("out"));
    Value conv2d = rewriter
                       .create<linalg::Conv2DNchwFchwOp>(
                           loc, out.getType(), ValueRange{paddedInput, weight},
                           out, stridesAttr, dilationAttr, variant)
                       .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv2d);
    //    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType,
    //    conv2d);
    return success();
  }
};
} // namespace

namespace {

struct ConvertElementwiseOp : ConversionPattern {
  ConvertElementwiseOp(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<AtenReluOp>(op))
      return rewriter.notifyMatchFailure(op, "not a supported elementwise op");

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    Location loc = op->getLoc();
    auto tensorOperands = llvm::to_vector<6>(llvm::make_filter_range(
        operands, [](Value v) { return v.getType().isa<RankedTensorType>(); }));
    auto resultType = getTypeConverter()
                          ->convertType(op->getResult(0).getType())
                          .cast<RankedTensorType>();
    auto resultRank = resultType.getRank();

    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/1);
    SmallVector<Value> resultShape(resultRank, c1);
    SmallVector<AffineMap> indexingMaps;
    for (Value tensorOperand : tensorOperands) {
      SmallVector<AffineExpr> exprs;
      auto type = tensorOperand.getType().cast<RankedTensorType>();
      for (auto size : llvm::enumerate(type.getShape())) {
        if (size.value() == 1) {
          exprs.push_back(rewriter.getAffineConstantExpr(0));
          continue;
        }
        auto resultDim = size.index() + (resultRank - type.getRank());
        exprs.push_back(rewriter.getAffineDimExpr(resultDim));
        auto currentDimSize =
            rewriter.create<tensor::DimOp>(loc, tensorOperand, size.index());
        if (resultShape[resultDim] == c1) {
          resultShape[resultDim] = currentDimSize;
          continue;
        }
        auto equalToRunning = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, resultShape[resultDim],
            currentDimSize);
        rewriter.create<AssertOp>(loc, equalToRunning,
                                  "mismatched size for broadcast");
      }
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, getContext()));
    }

    SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("inplace"));
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultShape, resultType.getElementType());
    bool hadErrorCreatingPayload = false;
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/initTensor.getType(),
        /*inputs=*/tensorOperands,
        /*outputs=*/initTensor,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
          Value result = createLinalgPayloadCalculationForElementwiseOp(
              b, loc, getTypeConverter(), payloadArgs, op, operands);
          if (!result) {
            hadErrorCreatingPayload = true;
            return;
          }
          b.create<linalg::YieldOp>(loc, result);
        },
        variant);

    if (hadErrorCreatingPayload)
      return failure();
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultType,
                                                generic.getResult(0));
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMaxPool2dWithIndicesOutOp
    : public OpConversionPattern<AtenMaxPool2dWithIndicesOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMaxPool2dWithIndicesOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value self = adaptor.self();
    Value ceilMode = adaptor.ceil_mode();

    Type elementType = self.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    SmallVector<int64_t, 2> strideInts;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    SmallVector<int64_t, 2> paddingInts;
    /**/ if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int paddings");
    SmallVector<int64_t, 2> kernelSizeInts;
    if (!matchPattern(op.kernel_size(), m_TorchConstantIntList(kernelSizeInts)))
      return rewriter.notifyMatchFailure(op, "only support kernel size ints");

    Value falseValue = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(rewriter.getIntegerType(1), 0));
    Value ceilModeFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, ceilMode, falseValue);
    rewriter.create<AssertOp>(
        loc, ceilModeFalse,
        rewriter.getStringAttr("only ceil_mode false is supported"));

    //    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    //    paddingIncludingNC.insert(paddingIncludingNC.end(),
    //    paddingInts.begin(),
    //                              paddingInts.end());
    //    Value paddedInput = getPaddedTensor(op, rewriter, self,
    //    paddingIncludingNC);

    //    Value N = getDimOp(rewriter, loc, self, 0);
    //    Value C = getDimOp(rewriter, loc, self, 1);
    //    Value H = getDimOp(rewriter, loc, self, 2);
    //    Value W = getDimOp(rewriter, loc, self, 3);
    //
    //    SmallVector<Value> paddingIntValues =
    //        getAsConstantIntValues(rewriter, loc, paddingInts);
    //    SmallVector<Value> dilationIntValues =
    //        getAsConstantIntValues(rewriter, loc, dilationInts);
    //    SmallVector<Value> kernelSizeIntValues =
    //        getAsConstantIntValues(rewriter, loc, kernelSizeInts);
    //    SmallVector<Value> strideIntValues =
    //        getAsConstantIntValues(rewriter, loc, strideInts);

    //    Value Hout = getOutputDimForConvOps(
    //        rewriter, loc, H, paddingIntValues[0], dilationIntValues[0],
    //        kernelSizeIntValues[0], strideIntValues[0]);
    //    Value Wout = getOutputDimForConvOps(
    //        rewriter, loc, W, paddingIntValues[1], dilationIntValues[1],
    //        kernelSizeIntValues[1], strideIntValues[1]);

    //    auto initialAttr = rewriter.getFloatAttr(
    //        elementType,
    //        APFloat::getSmallest(
    //            elementType.cast<mlir::FloatType>().getFloatSemantics(),
    //            /*Negative*/ true));
    auto stridesAttr = rewriter.getI64VectorAttr(strideInts);
    auto dilationAttr = rewriter.getI64VectorAttr(dilationInts);
    Value windowTensor_ = createGetTensorLiteral(rewriter, loc, kernelSizeInts,
                                                 0.0f, elementType);

    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("out"));
    ArrayRef<NamedAttribute> attributes = {variant};
    Value out = adaptor.out();
    Value maxPool2d =
        rewriter
            .create<linalg::PoolingNchwMaxOp>(
                loc, out.getType(), ValueRange{self, windowTensor_}, out,
                stridesAttr, dilationAttr, attributes)
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    Value res = rewriter.create<tensor::CastOp>(loc, newResultType, maxPool2d)
                    .getResult();
    rewriter.replaceOp(op, {res, op.indices()});
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenAdaptiveAvgPool2dOutOp
    : public OpConversionPattern<AtenAdaptiveAvgPool2dOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenAdaptiveAvgPool2dOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.self(); /* in form of N*C*H*W */
    Value out = adaptor.out();    /* in form of N*C*H*W */
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    Type elementType = inputType.getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");

    auto inputRank = inputType.getRank();
    if (inputRank != 4)
      return rewriter.notifyMatchFailure(op, "input should be rank 4");

    SmallVector<int64_t, 2> expects{1, 1};
    // Pattern match against the op's original operands, because otherwise we
    // will get the lowered version of the operands which is harder to pattern
    // match.
    if (!isConstantIntListMatching(op.output_size(), expects))
      return rewriter.notifyMatchFailure(
          op, "only support output_size with H and W both equal to constant 1");

    Value N = getDimOp(rewriter, loc, input, 0);
    Value C = getDimOp(rewriter, loc, input, 1);
    // TODO: batch size and channels
    auto val = createGetTensorLiteral(rewriter, loc, {7, 2}, 0.0f, elementType);
    //    Value initTensor = rewriter.create<linalg::InitTensorOp>(
    //        loc, ValueRange{N, C}, elementType);
    //    Value c0 = rewriter.create<arith::ConstantOp>(
    //        loc, FloatAttr::get(elementType, 0.0));
    //    Value initTensor0 =
    //        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);

    SmallVector<AffineExpr, 2> ncExprs;
    ncExprs.push_back(mlir::getAffineDimExpr(0, context));
    ncExprs.push_back(mlir::getAffineDimExpr(1, context));
    auto ncIndexingMap = AffineMap::get(
        /*dimCount=*/4,
        /*symbolCount=*/0, ncExprs, context);
    SmallVector<AffineMap, 2> indexingMaps = {
        rewriter.getMultiDimIdentityMap(4), // input
        ncIndexingMap,                      // output
    };
    SmallVector<StringRef, 4> iteratorTypesSum{"parallel", "parallel",
                                               "reduction", "reduction"};
    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("out"));
    Value sumPool2d = rewriter
                          .create<linalg::GenericOp>(
                              loc, val.getType(), input, val,
                              /*indexingMaps=*/indexingMaps,
                              /*iteratorTypes=*/iteratorTypesSum,
                              [&](OpBuilder &b, Location loc, ValueRange args) {
                                Value input = args[0], sum = args[1];
                                Value result = rewriter.create<arith::AddFOp>(
                                    loc, sum, input);
                                b.create<linalg::YieldOp>(loc, result);
                              },
                              variant)
                          .getResult(0);

    // Calculate H*W so that avg can be got from sum / (H*W)
    Value H = getDimOp(rewriter, loc, input, 2);
    Value W = getDimOp(rewriter, loc, input, 3);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(
          loc, IntegerType::get(context, 64), v);
    };
    Value HtimesW = rewriter.create<arith::MulIOp>(loc, castIndexToInt(H),
                                                   castIndexToInt(W));
    Value HtimesWf =
        rewriter.create<arith::SIToFPOp>(loc, elementType, HtimesW);

    //    Value c1Index = rewriter.create<arith::ConstantIndexOp>(loc,
    //    /*value=*/1); Value outputTensor =
    //    rewriter.create<linalg::InitTensorOp>(
    // TODO: n ,c here
    //        loc, ValueRange{N, C, c1Index, c1Index}, elementType);
    SmallVector<AffineMap, 2> indexingMapsAvg{
        ncIndexingMap, rewriter.getMultiDimIdentityMap(4)};
    SmallVector<StringRef, 4> iteratorTypesAvg(4, "parallel");
    Value avgPool2d =
        rewriter
            .create<linalg::GenericOp>(
                loc, out.getType(), sumPool2d, out,
                /*indexingMaps=*/indexingMapsAvg,
                /*iteratorTypes=*/iteratorTypesAvg,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value avg = b.create<arith::DivFOp>(loc, args[0], HtimesWf);
                  b.create<linalg::YieldOp>(loc, avg);
                },
                variant)
            .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, avgPool2d);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenNativeBatchNormOutOp
    : public OpConversionPattern<AtenNativeBatchNormOutOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenNativeBatchNormOutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    Value runningMean = adaptor.running_mean();
    Value runningVar = adaptor.running_var();
    Value training = adaptor.training();
    Value eps = adaptor.eps();

    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();

    // TODO: Handle the None cases for the optional parameters:
    // weight, bias.
    if (failed(checkNotNone(rewriter, op, weight)) ||
        failed(checkNotNone(rewriter, op, bias)) ||
        failed(checkNotNone(rewriter, op, runningMean)) ||
        failed(checkNotNone(rewriter, op, runningVar)))
      return failure();

    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();
    auto runningMeanType = runningMean.getType().cast<RankedTensorType>();
    auto runningVarType = runningVar.getType().cast<RankedTensorType>();

    auto inputRank = inputType.getRank();
    if (inputRank <= 2)
      return rewriter.notifyMatchFailure(
          op, "input should have rank larger than 2");

    if (weightType.getRank() != 1 || biasType.getRank() != 1 ||
        runningMeanType.getRank() != 1 || runningVarType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expect weight, bias, running_mean and running_var to be rank 1");
    }

    // TODO: Add support for training.
    auto constFalse = rewriter.create<arith::ConstantOp>(
        loc, IntegerAttr::get(IntegerType::get(context, 1), 0));
    auto trainingFalse = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, training, constFalse);
    rewriter.create<AssertOp>(
        loc, trainingFalse,
        rewriter.getStringAttr("training is not supported for now"));

    // num_features – C from an expected input of size (N,C,D,H,W ...)
    Value numFeatures = rewriter.create<tensor::DimOp>(loc, input, 1);
    auto contractingDim0EqualsNumFeatures = [&](Value v) {
      auto dim0 = rewriter.create<tensor::DimOp>(loc, v, 0);
      auto dim0Equal = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, numFeatures, dim0);
      rewriter.create<AssertOp>(
          loc, dim0Equal,
          rewriter.getStringAttr(
              "expect the size of dim 0 equal to the number of features"));
    };
    contractingDim0EqualsNumFeatures(weight);
    contractingDim0EqualsNumFeatures(bias);
    contractingDim0EqualsNumFeatures(runningMean);
    contractingDim0EqualsNumFeatures(runningVar);

    auto indexingMap = AffineMap::get(
        /*dimCount=*/inputRank,
        /*symbolCount=*/0, rewriter.getAffineDimExpr(1), context);
    SmallVector<AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(inputRank), // input
        indexingMap,                                // weight
        indexingMap,                                // bias
        indexingMap,                                // runningMean
        indexingMap,                                // runningVar
        rewriter.getMultiDimIdentityMap(inputRank), // output
    };
    SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr("out"));
    Value batchNorm =
        rewriter
            .create<linalg::GenericOp>(
                loc, input.getType(),
                ValueRange{input, weight, bias, runningMean, runningVar}, input,
                /*indexingMaps=*/indexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [&](OpBuilder &b, Location loc, ValueRange args) {
                  Value input = args[0], weight = args[1], bias = args[2],
                        mean = args[3], var = args[4];
                  Value result = createLinalgPayloadCalculationForNormOps(
                      b, loc, var.getType(), input, mean, var, eps, weight,
                      bias);
                  b.create<linalg::YieldOp>(loc, result);
                }, variant)
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType(0));
    Value res = rewriter.create<tensor::CastOp>(loc, newResultType, batchNorm)
                    .getResult();
//    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, batchNorm);
    rewriter.replaceOp(op, {res, op.running_mean(), op.running_var()});
    return success();
  }
};
} // namespace

namespace {
class HLSConvertTorchToLinalg
    : public HLSConvertTorchToLinalgBase<HLSConvertTorchToLinalg> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect,
                           mlir::torch::Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<AtenMmOutOp>();
    patterns.add<ConvertAtenMmOutOp>(typeConverter, context);
    target.addIllegalOp<AtenMatmulOutOp>();
    patterns.add<ConvertAtenMatmulOutOp>(typeConverter, context);
    target.addIllegalOp<AtenTHNNConv2dOutOp>();
    patterns.add<ConvertAtenTHNNConv2dOutOp>(typeConverter, context);
    target.addIllegalOp<AtenReluOp>();
    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    target.addIllegalOp<AtenMaxPool2dWithIndicesOutOp>();
    patterns.add<ConvertAtenMaxPool2dWithIndicesOutOp>(typeConverter, context);
    target.addIllegalOp<AtenAdaptiveAvgPool2dOutOp>();
    patterns.add<ConvertAtenAdaptiveAvgPool2dOutOp>(typeConverter, context);
    target.addIllegalOp<AtenNativeBatchNormOutOp>();
    patterns.add<ConvertAtenNativeBatchNormOutOp>(typeConverter, context);

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
