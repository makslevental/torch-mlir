#include "BraggHLSPassDetail.h"
#include "BraggHLSPasses.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::linalg;
using namespace mlir::BraggHLS;
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
  b.create<RuntimeAssertOp>(
      loc, contractingDimEqual,
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
  Type ranked4DTensorType = tensor::PadOp::inferResultType(
      input.getType().cast<RankedTensorType>(), paddingInts, paddingInts);
  Value paddedInput = tensor::createPadScalarOp(
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

static Value convertScalarToDtype(OpBuilder &b, Location loc, Value scalar,
                                  Type dtype) {
  Type scalarType = scalar.getType();
  if (scalarType == dtype)
    return scalar;

  // TODO: For the byte(ui8) or char(i8) case, we need the unconverted dtype to
  // be able to know if we need signed or unsigned conversion.
  auto isByteOrChar = [](Type type) {
    if (auto integerTy = type.dyn_cast<mlir::IntegerType>()) {
      return integerTy.getWidth() == 8;
    }
    return false;
  };

  if (isByteOrChar(scalarType) || isByteOrChar(dtype) ||
      dtype.isSignlessInteger(1)) {
    // TODO: Handle to-boolean conversion(from-boolean conversion is handled).
    mlir::emitError(loc)
        << "unsupported byte, char or bool type for convertScalarToDtype "
        << scalarType << "(scalar type) -> " << dtype << "(dtype)";
    return nullptr;
  }

  if (auto dtypeFloat = dtype.dyn_cast<mlir::FloatType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>()) {
      if (scalarFloat.getWidth() > dtypeFloat.getWidth())
        return b.create<arith::TruncFOp>(loc, dtype, scalar);
      // Only scalarFloat width < dtypeFloat width can reach here.
      return b.create<arith::ExtFOp>(loc, dtype, scalar);
    }
    assert(scalarType.isa<mlir::IntegerType>());
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::UIToFPOp>(loc, dtype, scalar);
    // It's safe to use SIToFPOp because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::SIToFPOp>(loc, dtype, scalar);
  }

  if (auto dtypeInteger = dtype.dyn_cast<mlir::IntegerType>()) {
    if (auto scalarFloat = scalarType.dyn_cast<mlir::FloatType>())
      return b.create<arith::FPToSIOp>(loc, dtype, scalar);
    assert(scalarType.isa<mlir::IntegerType>());
    auto scalarInteger = scalarType.cast<mlir::IntegerType>();
    if (scalarInteger.getWidth() > dtypeInteger.getWidth())
      return b.create<arith::TruncIOp>(loc, dtype, scalar);
    if (scalarType.isSignlessInteger(1))
      return b.create<arith::ExtUIOp>(loc, dtype, scalar);
    // Only scalarInteger width < dtypeInteger width can reach here.
    // It's safe to use ExtSIOp here because ui8/si8 are the only ones where
    // unsigned handling is needed, and we checked for that case above.
    return b.create<arith::ExtSIOp>(loc, dtype, scalar);
  }

  llvm_unreachable("convertScalarToDtype should handle all the types");
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
    return b.create<arith::SelectOp>(loc, pred, payloadArgs[0], constZero);
  }
  op->emitError("unimplemented lowering in "
                "createLinalgPayloadCalculationForElementwiseOp");
  return nullptr;
}


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
        //        rewriter.create<cf::AssertOp>(loc, equalToRunning,
        //                                  "mismatched size for broadcast");
      }
      indexingMaps.push_back(AffineMap::get(
          /*dimCount=*/resultRank, /*symbolCount=*/0, exprs, getContext()));
    }

    SmallVector<StringRef> iteratorTypes(resultRank, "parallel");
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(resultRank));

    std::string variant_;
    Value resultTensor;
    if (isa<AtenReluOp>(op)) {
      variant_ = "inplace";
      resultTensor = rewriter.create<linalg::InitTensorOp>(
          loc, resultShape, resultType.getElementType());
    } else {
      return rewriter.notifyMatchFailure(op, "not a supported elementwise op");
    }
    NamedAttribute variant(rewriter.getStringAttr("variant"),
                           rewriter.getStringAttr(variant_));
    bool hadErrorCreatingPayload = false;
    auto generic = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/resultTensor.getType(),
        /*inputs=*/tensorOperands,
        /*outputs=*/resultTensor,
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
class HLSConvertTorchToLinalg
    : public BraggHLSConvertTorchToLinalgBase<HLSConvertTorchToLinalg> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect,
                           math::MathDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect,
                           mlir::torch::Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    //    typeConverter.addArgumentMaterialization(
    //        [](OpBuilder &builder, Torch::BaseTensorType type, ValueRange
    //        inputs,
    //           Location loc) -> Value {
    //          assert(inputs.size() == 1);
    //          assert(inputs[0].getType().isa<BaseTensorType>());
    //          return copyTensorToType(builder, loc, type, inputs[0]);
    //        });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    //    target.addIllegalOp<AtenMmOutOp>();
    //    patterns.add<ConvertAtenMmOutOp>(typeConverter, context);
    //    target.addIllegalOp<AtenMatmulOutOp>();
    //    patterns.add<ConvertAtenMatmulOutOp>(typeConverter, context);
    //    target.addIllegalOp<AtenTHNNConv2dOutOp>();
    //    patterns.add<ConvertAtenTHNNConv2dOutOp>(typeConverter, context);
    //    target.addIllegalOp<AtenReluOp, AtenAddOutTensorOp>();
    //    patterns.add<ConvertElementwiseOp>(typeConverter, context);
    //    target.addIllegalOp<AtenMaxPool2dWithIndicesOutOp>();
    //    patterns.add<ConvertAtenMaxPool2dWithIndicesOutOp>(typeConverter,
    //    context); target.addIllegalOp<AtenAdaptiveAvgPool2dOutOp>();
    //    patterns.add<ConvertAtenAdaptiveAvgPool2dOutOp>(typeConverter,
    //    context); target.addIllegalOp<AtenNativeBatchNormOutOp>();
    //    patterns.add<ConvertAtenNativeBatchNormOutOp>(typeConverter, context);
    //    target.addIllegalOp<AtenLinearOutOp>();
    //    patterns.add<ConvertAtenLinearOutOp>(typeConverter, context);
    //    target.addIllegalOp<AtenPermuteOp>();
    //    patterns.add<ConvertAtenPermuteOp>(typeConverter, context);
    //    target.addIllegalOp<Aten_ConvolutionOp>();
    //    patterns.add<ConvertAten_ConvolutionOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::BraggHLS::createBraggHLSConvertTorchToLinalgPass() {
  return std::make_unique<HLSConvertTorchToLinalg>();
}
