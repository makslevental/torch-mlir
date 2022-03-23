//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class ConvertAtenMmOp : public OpConversionPattern<AtenMmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.mat2();

    // A user can write an errorneous program where `aten.mm` is in fact called
    // with operands of invalid rank or dtype. We cannot convert to linalg in
    // this case or we will get a verifier error, which corresponds to breaking
    // of *internal* compiler invariants, and for a user manifests as a compiler
    // crash in the worst case (such as we try to canonicalize/fold/print the
    // invalid op before the verifier gets to see it -- also release builds of a
    // mature compiler usually have the verifier turned off for compile time
    // reasons).
    //
    // The compiler cannot crash even if the user wrote an erroneous program!
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
    rewriter.create<cf::AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for torch.aten.mm"));

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ValueRange{lhsDim0, rhsDim1}, elementType);
    Value c0 = rewriter.create<arith::ConstantOp>(
        loc, FloatAttr::get(elementType, 0.0));
    Value zeroFill =
        rewriter.create<linalg::FillOp>(loc, c0, initTensor).getResult(0);
    Value matmul = rewriter
                       .create<linalg::MatmulOp>(loc, zeroFill.getType(),
                                                 ValueRange{lhs, rhs}, zeroFill)
                       .getResult(0);
    // When constructed with just dynamic sizes, InitTensorOp will have a result
    // type which has all `?`'s for dimensions, which might not be the result
    // type of `op`. The constraints on later linalg ops means that the result
    // of the MatmulOp will have this type too. So cast it to the desired type
    // so that in the end we have the original result type.
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenMatmulOp : public OpConversionPattern<AtenMatmulOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.other();

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

      Value zeroTensor = createZeroInitTensor(rewriter, loc, {}, elementType);
      Value dotProd =
          rewriter
              .create<linalg::DotOp>(loc, zeroTensor.getType(),
                                     ValueRange{lhs, rhs}, zeroTensor)
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

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{rhsDim1}, elementType);
      Value matmul =
          rewriter
              .create<linalg::VecmatOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
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

      Value zeroTensor =
          createZeroInitTensor(rewriter, loc, ValueRange{lhsDim0}, elementType);
      Value matmul =
          rewriter
              .create<linalg::MatvecOp>(loc, zeroTensor.getType(),
                                        ValueRange{lhs, rhs}, zeroTensor)
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

      Value initTensor0 =
          createZeroInitTensor(rewriter, loc, resultShape, elementType);

      auto indexingMaps =
          AffineMap::inferFromExprList({lhsExpr, rhsExpr, outExpr});
      iteratorTypes.insert(iteratorTypes.end(),
                           {"parallel", "reduction", "parallel"});

      Value finalRes =
          rewriter
              .create<linalg::GenericOp>(
                  loc, initTensor0.getType(), ValueRange{lhs, rhs}, initTensor0,
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
class ConvertAtenBmmOp : public OpConversionPattern<AtenBmmOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenBmmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    Location loc = op->getLoc();
    Value lhs = adaptor.self();
    Value rhs = adaptor.mat2();
    RankedTensorType lhsType = lhs.getType().cast<RankedTensorType>();
    RankedTensorType rhsType = rhs.getType().cast<RankedTensorType>();

    if (lhsType.getRank() != 3 || rhsType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected both operands to aten.bmm to be rank 3");
    }
    if (!lhsType.getElementType().isa<mlir::FloatType>() ||
        lhsType.getElementType() != rhsType.getElementType())
      return op.emitError(
          "unimplemented: non floating point operands or operands of "
          "different types");

    Value lhsDim0 = getDimOp(rewriter, loc, lhs, 0);
    Value lhsDim1 = getDimOp(rewriter, loc, lhs, 1);
    Value lhsDim2 = getDimOp(rewriter, loc, lhs, 2);
    Value rhsDim0 = getDimOp(rewriter, loc, rhs, 0);
    Value rhsDim1 = getDimOp(rewriter, loc, rhs, 1);
    Value rhsDim2 = getDimOp(rewriter, loc, rhs, 2);

    // Check the batch numbers are equal.
    checkDimEqualHelper(rewriter, loc, lhsDim0, rhsDim0);

    // Check the matrixs shapes are valid for mulplication.
    checkDimEqualHelper(rewriter, loc, lhsDim2, rhsDim1);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    Type elementType = newResultType.cast<TensorType>().getElementType();
    Value initTensor0 = createZeroInitTensor(
        rewriter, loc, ValueRange{lhsDim0, lhsDim1, rhsDim2}, elementType);

    Value bmm =
        rewriter
            .create<linalg::BatchMatmulOp>(loc, initTensor0.getType(),
                                           ValueRange{lhs, rhs}, initTensor0)
            .getResult(0);
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, bmm);
    return success();
  }
};
} // namespace

namespace {
// See comments at in convertMmOp and the heading for this section for general
// considerations. This function needs to be auto-generated.
class ConvertAtenLinearOp : public OpConversionPattern<AtenLinearOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *context = op->getContext();
    Location loc = op->getLoc();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    Value bias = adaptor.bias();
    // TODO: Handle the case of bias being None (bias is optional).
    if (failed(verifyLinalgCompatibleTypes(op, rewriter)))
      return failure();
    auto inputType = input.getType().cast<RankedTensorType>();
    auto weightType = weight.getType().cast<RankedTensorType>();
    auto biasType = bias.getType().cast<RankedTensorType>();

    if (inputType.getRank() != 2 && inputType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          op, "expected  input to be rank 2 or rank 3");
    }

    // Only handle the case of rank 2 `weight` for now.
    // TODO: Insert the appropriate reshape to collapse any leading dimensions.
    if (weightType.getRank() != 2 || biasType.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected weight to be rank 2 and bias to be rank 1");
    }
    // TODO: Handle type promotion. What are ATen's promotion rules?
    if (inputType.getElementType() != weightType.getElementType() ||
        inputType.getElementType() != biasType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: type promotion");
    }

    // TODO: We can handle a static size 1 here at some complexity cost, but the
    // dynamic case is not representable in linalg. We don't handle either for
    // now. Biases are generally statically shaped for most models (since for
    // inference they are constants, and for training they don't change shape
    // typically), so this is not too constraining.
    auto biasSize = bias.getType().cast<RankedTensorType>().getShape()[0];
    if (biasSize == 1 || biasSize == ShapedType::kDynamicSize)
      return rewriter.notifyMatchFailure(
          op, "unimplemented: size-1 broadcasting for aten::LinearOp");

    Value batchDim = nullptr;
    int restDim = 0;
    if (inputType.getRank() == 3) {
      batchDim = getDimOp(rewriter, loc, input, 0);
      restDim = 1;
    }

    Value inputDim0 = getDimOp(rewriter, loc, input, restDim + 0);
    Value inputDim1 = getDimOp(rewriter, loc, input, restDim + 1);
    Value weightDim0 = getDimOp(rewriter, loc, weight, 0);
    Value weightDim1 = getDimOp(rewriter, loc, weight, 1);
    Value biasDim0 = getDimOp(rewriter, loc, bias, 0);
    Value contractingDimEqual = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, inputDim1, weightDim1);
    rewriter.create<cf::AssertOp>(
        loc, contractingDimEqual,
        rewriter.getStringAttr(
            "mismatching contracting dimension for aten.linear"));
    // Here we take advantage of ruling out the size-1 case above.
    // In the static-size-1 case, we will not emit this check at all.
    Value biasSizeCorrect = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, weightDim0, biasDim0);
    rewriter.create<cf::AssertOp>(
        loc, biasSizeCorrect,
        rewriter.getStringAttr("mismatching bias size for aten.linear"));

    Value initTensor;
    SmallVector<AffineMap> broadcastIndexingMaps;
    Value transposedWeightInitTensor;
    if (inputType.getRank() > 2) {
      initTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{batchDim, inputDim0, weightDim0},
          inputType.getElementType());
      transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{batchDim, weightDim1, weightDim0},
          weightType.getElementType());
      broadcastIndexingMaps = {
          AffineMap::get(
              /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
              {rewriter.getAffineDimExpr(1 + restDim)}, context),
          rewriter.getMultiDimIdentityMap(inputType.getRank())};
    } else {
      initTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{inputDim0, weightDim0},
          inputType.getElementType());
      transposedWeightInitTensor = rewriter.create<linalg::InitTensorOp>(
          loc, ValueRange{weightDim1, weightDim0}, weightType.getElementType());
      broadcastIndexingMaps = {
          AffineMap::get(
              /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
              {rewriter.getAffineDimExpr(1)}, context),
          rewriter.getMultiDimIdentityMap(inputType.getRank())};
    }

    SmallVector<StringRef> iteratorTypes(inputType.getRank(), "parallel");
    Value broadcasted =
        rewriter
            .create<linalg::GenericOp>(
                loc, initTensor.getType(), bias, initTensor,
                /*indexingMaps=*/broadcastIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    // We need a matmul with dimension ordering (N, K) * (M, K), so transpose
    // the weights to fit into linalg::MatmulOp which is (N, K) * (K, M).
    // TODO: This whole aten.linear lowering should eventually be generated from
    // a single linalg ODS generator statement. Both the bias and matmul part.
    SmallVector<AffineMap> transposeIndexingMaps = {
        AffineMap::get(
            /*dimCount=*/inputType.getRank(), /*symbolCount=*/0,
            {rewriter.getAffineDimExpr(1 + restDim),
             rewriter.getAffineDimExpr(0 + restDim)},
            context),
        rewriter.getMultiDimIdentityMap(inputType.getRank())};
    Value transposedWeights =
        rewriter
            .create<linalg::GenericOp>(
                loc, transposedWeightInitTensor.getType(), weight,
                transposedWeightInitTensor,
                /*indexingMaps=*/transposeIndexingMaps,
                /*iteratorTypes=*/iteratorTypes,
                [](OpBuilder &b, Location loc, ValueRange args) {
                  b.create<linalg::YieldOp>(loc, args[0]);
                })
            .getResult(0);
    Value matmul;
    if (batchDim)
      matmul = rewriter
                   .create<linalg::BatchMatmulOp>(
                       loc, broadcasted.getType(),
                       ValueRange{input, transposedWeights}, broadcasted)
                   .getResult(0);
    else
      matmul = rewriter
                   .create<linalg::MatmulOp>(
                       loc, broadcasted.getType(),
                       ValueRange{input, transposedWeights}, broadcasted)
                   .getResult(0);

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, matmul);
    return success();
  }
};
} // namespace

namespace {
Value flipTo(ConversionPatternRewriter &rewriter, Location &loc, Value tensor,
             uint64_t fromDim, uint64_t toDim) {
  RankedTensorType tType = tensor.getType().cast<RankedTensorType>();
  SmallVector<Value> tSizes = getTensorSizes(rewriter, loc, tensor);

  Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  SmallVector<Value> flip;
  for (size_t i = 0; i < tSizes.size(); i++)
    flip.push_back(rewriter.create<arith::SubIOp>(loc, tSizes[i], one));

  SmallVector<StringRef> iteratorTypes(tSizes.size(), "parallel");
  SmallVector<AffineExpr> exprs;
  for (size_t i = 0; i < tSizes.size(); i++)
    exprs.push_back(rewriter.getAffineDimExpr(i));
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList({exprs, exprs});
  Value outTensor = rewriter.create<linalg::InitTensorOp>(
      loc, tSizes, tType.getElementType());

  // cast here using tType.getShape()
  Value refinedShapeOutTensor = rewriter.create<tensor::CastOp>(loc, tType, outTensor);

  return rewriter
      .create<linalg::GenericOp>(
          loc, tType, tensor, outTensor, indexingMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            SmallVector<Value> indices;
            for (size_t i = 0; i < tSizes.size(); i++) {
              if (i >= fromDim && i < toDim) {
                Value index = b.create<linalg::IndexOp>(loc, i);
                indices.push_back(b.create<arith::SubIOp>(loc, flip[i], index));
                continue;
              }
              indices.push_back(b.create<linalg::IndexOp>(loc, i));
            }
            std::swap(indices[0], indices[1]);
            b.create<linalg::YieldOp>(
                loc, rewriter
                         .create<tensor::ExtractOp>(loc, tType.getElementType(),
                                                    tensor, indices)
                         .getResult());
          })
      .getResult(0);
}

Value rotate(ConversionPatternRewriter &rewriter, Location &loc, Value tensor) {
  RankedTensorType tType = tensor.getType().cast<RankedTensorType>();
  SmallVector<Value> tSizes = getTensorSizes(rewriter, loc, tensor);

  SmallVector<StringRef> iteratorTypes(tSizes.size(), "parallel");
  SmallVector<AffineExpr> inExpr{rewriter.getAffineDimExpr(0),
                                 rewriter.getAffineDimExpr(1)};
  SmallVector<AffineExpr> outExpr{rewriter.getAffineDimExpr(1),
                                  rewriter.getAffineDimExpr(0)};
  for (size_t i = 2; i < tSizes.size(); i++) {
    inExpr.push_back(rewriter.getAffineDimExpr(i));
    outExpr.push_back(rewriter.getAffineDimExpr(i));
  }
  SmallVector<AffineMap> indexingMaps =
      AffineMap::inferFromExprList({inExpr, outExpr});
  Value outTensor = rewriter.create<linalg::InitTensorOp>(
      loc, tSizes, tType.getElementType());

//   cast here using tType.getShape()
  Value refinedShapeOutTensor = rewriter.create<tensor::CastOp>(loc, tType, outTensor);

  return rewriter
      .create<linalg::GenericOp>(
          loc, tType, tensor, refinedShapeOutTensor, indexingMaps, iteratorTypes,
          [&](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
          })
      .getResult(0);
}

class ConvertAtenConvolutionBackwardOp
    : public OpConversionPattern<AtenConvolutionBackwardOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConvolutionBackwardOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();

    Value grad = adaptor.grad_output();
    Value input = adaptor.input();
    Value weight = adaptor.weight();
    // Value bias_sizes= adaptor.bias_sizes();
    // Value padding = adaptor.padding();
    // Value transposed = adaptor.transposed();
    // Value output_padding = adaptor.output_padding();
    // Value groups = adaptor.groups();
    // Value output_mask = adaptor.output_mask();

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, intType, v);
    };
    Type eType = input.getType().cast<RankedTensorType>().getElementType();
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(eType, 0.0));

    SmallVector<Value> gradSizes = getTensorSizes(rewriter, loc, grad);
    size_t gradRank = gradSizes.size();
    SmallVector<Value> inputSizes = getTensorSizes(rewriter, loc, input);
    size_t inputRank = inputSizes.size();
    SmallVector<Value> weightSizes = getTensorSizes(rewriter, loc, weight);
    size_t weightRank = weightSizes.size();

    SmallVector<int64_t> stride;
    if (!matchPattern(adaptor.stride(), m_TorchConstantIntList(stride)))
      return rewriter.notifyMatchFailure(op, "only constant strides supported");
    SmallVector<int64_t> dilation;
    if (!matchPattern(adaptor.dilation(), m_TorchConstantIntList(dilation)))
      return rewriter.notifyMatchFailure(op,
                                         "only constant dilation supported");
    SmallVector<Value> output_mask_val;
    if (!getListConstructElements(adaptor.output_mask(), output_mask_val))
      return rewriter.notifyMatchFailure(op,
                                         "only constant output mask supported");
    SmallVector<bool> output_mask;
    for (Value out : output_mask_val) {
      bool mask_val;
      if (!matchPattern(out, m_TorchConstantBool(&mask_val)))
        return rewriter.notifyMatchFailure(
            op, "only constant boolean output mask values supported");
      output_mask.push_back(mask_val);
    }
    SmallVector<int64_t> padding;
    if (!matchPattern(adaptor.padding(), m_TorchConstantIntList(padding)))
      return rewriter.notifyMatchFailure(op, "only constant padding supported");
    SmallVector<Value> paddingVals;
    for (auto p : padding)
      paddingVals.push_back(rewriter.create<arith::ConstantOp>(
          loc, IntegerAttr::get(intType, p)));
    SmallVector<Value> dilationVals;
    for (auto d : dilation)
      dilationVals.push_back(rewriter.create<arith::ConstantOp>(
          loc, IntegerAttr::get(intType, d)));
    SmallVector<Value> strideVals;
    for (auto s : stride)
      strideVals.push_back(rewriter.create<arith::ConstantOp>(
          loc, IntegerAttr::get(intType, s)));

    auto strideAttr = rewriter.getI64VectorAttr(stride);
    auto dilationAttr = rewriter.getI64VectorAttr(dilation);

    // Ensure static info is correct (i.e rank of inputs, types, etc)
    if (gradRank != inputRank)
      return rewriter.notifyMatchFailure(
          op, "invalid input - grad_output and input must have the same rank");

    SmallVector<Value> kernelSizes;
    for (auto i = weightSizes.begin() + 2; i != weightSizes.end(); i++)
      kernelSizes.push_back(*i);

    // Add asserts for dynamic info (matching sizes + strides, etc)
    // TODO

    Value kernel = flipTo(rewriter, loc, weight, 2, weightRank);
    Value gradFlip = rotate(rewriter, loc, grad);
    SmallVector<Value> gradFlipSizes = getTensorSizes(rewriter, loc, gradFlip);

    // Pad input
    SmallVector<Value> resInputSizes{inputSizes[0], inputSizes[1]};
    SmallVector<Value> resWeightSizes{weightSizes[0], weightSizes[1]};
    for (size_t i = 2; i < weightRank; i++) {
      resInputSizes.push_back(torch_to_linalg::getOutputDimForConvOps(
          rewriter, loc, inputSizes[i], paddingVals[i - 2], dilationVals[i - 2],
          castIndexToInt(weightSizes[i]), strideVals[i - 2]));
      resWeightSizes.push_back(torch_to_linalg::getOutputDimForConvOps(
          rewriter, loc, gradSizes[i], paddingVals[i - 2], dilationVals[i - 2],
          castIndexToInt(inputSizes[i]), strideVals[i - 2]));
    }

    Value resInputTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resInputSizes, eType);
    resInputTensor =
        rewriter.create<linalg::FillOp>(loc, zero, resInputTensor).getResult(0);
    Value resWeightTensor =
        rewriter.create<linalg::InitTensorOp>(loc, resWeightSizes, eType);
    resWeightTensor =
        rewriter.create<linalg::FillOp>(loc, zero, resWeightTensor)
            .getResult(0);

    if (gradRank == 4) { // 2d convolution
      resInputTensor =
          rewriter
              .create<linalg::Conv2DNchwFchwOp>(
                  loc, resInputTensor.getType(), ValueRange{grad, kernel},
                  ValueRange{resInputTensor}, strideAttr, dilationAttr)
              .getResult(0);
      resWeightTensor =
          rewriter
              .create<linalg::Conv2DNchwFchwOp>(
                  loc, resWeightTensor.getType(),
                  ValueRange{gradFlip, rotate(rewriter, loc, input)},
                  ValueRange{resWeightTensor}, strideAttr, dilationAttr)
              .getResult(0);
    } else {
      return rewriter.notifyMatchFailure(
          op, "invalid input: only 2D convolutions are supported");
    }

    // Sum up grad output (the bias gradient)
    DenseSet<int64_t> dimSet;
    for (size_t i = 1; i < gradRank; i++)
      dimSet.insert(i);
    Value resBias = torch_to_linalg::createReductionLinalgGeneric(
        rewriter, loc, gradFlip, dimSet, false, zero,
        [&](OpBuilder &b, Location loc, ValueRange payloadArgs) {
          Value result = rewriter.create<arith::AddFOp>(loc, payloadArgs[0],
                                                        payloadArgs[1]);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, {resInputTensor, resWeightTensor, resBias});

    return success();
  }
};
} // namespace

namespace {
class ConvertAtenConvolutionOp : public OpConversionPattern<AtenConvolutionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value groups = adaptor.groups();

    Type elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    if (!elementType.isa<mlir::FloatType>())
      return op.emitError("unimplemented: non-floating point type");
    SmallVector<Value> inSizes = getTensorSizes(rewriter, loc, input);
    size_t inRank = inSizes.size();
    if(inRank != 4)
      return rewriter.notifyMatchFailure(op, "only 2D convolution currently supported");

    Type intType = IntegerType::get(context, 64);
    auto castIndexToInt = [&](Value v) {
      return rewriter.create<arith::IndexCastOp>(loc, intType, v);
    };

    SmallVector<int64_t> paddingInts;
    if (!matchPattern(op.padding(), m_TorchConstantIntList(paddingInts))) {
      return rewriter.notifyMatchFailure(
          op, "only support constant padding values");
    }
    SmallVector<int64_t> strideInts;
    if (!matchPattern(op.stride(), m_TorchConstantIntList(strideInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int strides");
    SmallVector<int64_t> dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");

    Value N = getDimOp(rewriter, loc, input, 0);
    SmallVector<Value> inDims;
    for(size_t i = 2; i < inRank; i++)
      inDims.push_back(getDimOp(rewriter, loc, input, i));
    Value F = getDimOp(rewriter, loc, weight, 0);
    SmallVector<Value> weightDims;
    for(size_t i = 2; i < inRank; i++)
      weightDims.push_back(getDimOp(rewriter, loc, weight, i));

    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value groupEqual1 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, groups, c1);
    rewriter.create<cf::AssertOp>(
        loc, groupEqual1, rewriter.getStringAttr("expect groups to be 1"));

    // Pad the input tensor according to padding.
    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput = torch_to_linalg::getPaddedTensor(op, rewriter, input,
                                                         paddingIncludingNC);

    SmallVector<Value> paddingIntValues =
        getAsConstantIntValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    SmallVector<Value> outDims{N, F};
    for(size_t i = 0; i < inRank-2; i++)
      outDims.push_back(torch_to_linalg::getOutputDimForConvOps(rewriter, loc, inDims[i], paddingIntValues[i], dilationIntValues[i], castIndexToInt(weightDims[i]), strideIntValues[i]));

    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, outDims, elementType);

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
    Value conv;
    // if(inRank == 3) {
    //   conv =
    //     rewriter
    //         .create<linalg::Conv1DNwcWcfOp>(
    //             loc, biasInitTensor.getType(), ValueRange{paddedInput, weight},
    //             biasInitTensor, stridesAttr, dilationAttr)
    //         .getResult(0);
    // } else if(inRank == 4) {
      conv =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, biasInitTensor.getType(), ValueRange{paddedInput, weight},
                biasInitTensor, stridesAttr, dilationAttr)
            .getResult(0);
    // } else if(inRank == 5) {
    //   conv =
    //     rewriter
    //         .create<linalg::Conv3DNdhwcDhwcfOp>(
    //             loc, biasInitTensor.getType(), ValueRange{paddedInput, weight},
    //             biasInitTensor, stridesAttr, dilationAttr)
    //         .getResult(0);
    // }

    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv);
    return success();
  }
};
} // namespace

namespace {
class ConvertAtenConv2dOp : public OpConversionPattern<AtenConv2dOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AtenConv2dOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    Value input = adaptor.input();   /* in form of N*C*H*W */
    Value weight = adaptor.weight(); /* in form of F*C*H*W */
    Value groups = adaptor.groups();

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
    SmallVector<int64_t, 2> dilationInts;
    if (!matchPattern(op.dilation(), m_TorchConstantIntList(dilationInts)))
      return rewriter.notifyMatchFailure(op,
                                         "only support constant int dilations");
    Value c1 =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(intType, 1));
    Value groupEqual1 = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, groups, c1);
    rewriter.create<cf::AssertOp>(
        loc, groupEqual1, rewriter.getStringAttr("expect groups to be 1"));

    // Pad the input tensor according to padding.
    SmallVector<int64_t, 4> paddingIncludingNC = {0, 0};
    paddingIncludingNC.insert(paddingIncludingNC.end(), paddingInts.begin(),
                              paddingInts.end());
    Value paddedInput = torch_to_linalg::getPaddedTensor(op, rewriter, input,
                                                         paddingIncludingNC);

    SmallVector<Value> paddingIntValues =
        getAsConstantIntValues(rewriter, loc, paddingInts);
    SmallVector<Value> dilationIntValues =
        getAsConstantIntValues(rewriter, loc, dilationInts);
    SmallVector<Value> strideIntValues =
        getAsConstantIntValues(rewriter, loc, strideInts);

    Value Hout = torch_to_linalg::getOutputDimForConvOps(
        rewriter, loc, Hin, paddingIntValues[0], dilationIntValues[0],
        castIndexToInt(weightH), strideIntValues[0]);
    Value Wout = torch_to_linalg::getOutputDimForConvOps(
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
    Value conv2d =
        rewriter
            .create<linalg::Conv2DNchwFchwOp>(
                loc, biasInitTensor.getType(), ValueRange{paddedInput, weight},
                biasInitTensor, stridesAttr, dilationAttr)
            .getResult(0);
    Type newResultType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<tensor::CastOp>(op, newResultType, conv2d);
    return success();
  }
};
} // namespace

void mlir::torch::torch_to_linalg::populateLinearPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenMmOp>();
  patterns.add<ConvertAtenMmOp>(typeConverter, context);
  target.addIllegalOp<AtenMatmulOp>();
  patterns.add<ConvertAtenMatmulOp>(typeConverter, context);
  target.addIllegalOp<AtenBmmOp>();
  patterns.add<ConvertAtenBmmOp>(typeConverter, context);
  target.addIllegalOp<AtenLinearOp>();
  patterns.add<ConvertAtenLinearOp>(typeConverter, context);
  target.addIllegalOp<AtenConv2dOp>();
  patterns.add<ConvertAtenConv2dOp>(typeConverter, context);
   target.addIllegalOp<AtenConvolutionBackwardOp>();
   patterns.add<ConvertAtenConvolutionBackwardOp>(typeConverter, context);
  target.addIllegalOp<AtenConvolutionOp>();
  patterns.add<ConvertAtenConvolutionOp>(typeConverter, context);
}
