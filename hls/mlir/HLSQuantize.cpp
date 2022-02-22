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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <iostream>

using namespace mlir;
using namespace mlir::torch::HLS;

// struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
//   ModuleOpConversion(MLIRContext *context, StringRef topLevelFunction,
//                      calyx::ProgramOp *programOpOutput)
//       : OpRewritePattern<mlir::ModuleOp>(context),
//         programOpOutput(programOpOutput), topLevelFunction(topLevelFunction)
//         {
//     assert(programOpOutput->getOperation() == nullptr &&
//            "this function will set programOpOutput post module conversion");
//   }
//
//   LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
//                                 PatternRewriter &rewriter) const override {
//     if (!moduleOp.getOps<calyx::ProgramOp>().empty())
//       return failure();
//
//     rewriter.updateRootInPlace(moduleOp, [&] {
//       // Create ProgramOp
//       rewriter.setInsertionPointAfter(moduleOp);
//       auto programOp = rewriter.create<calyx::ProgramOp>(
//           moduleOp.getLoc(), StringAttr::get(getContext(),
//           topLevelFunction));
//
//       // Inline the module body region
//       rewriter.inlineRegionBefore(moduleOp.getBodyRegion(),
//                                   programOp.getBodyRegion(),
//                                   programOp.getBodyRegion().end());
//
//       // Inlining the body region also removes ^bb0 from the module body
//       // region, so recreate that, before finally inserting the programOp
//       auto moduleBlock = rewriter.createBlock(&moduleOp.getBodyRegion());
//       rewriter.setInsertionPointToStart(moduleBlock);
//       rewriter.insert(programOp);
//       *programOpOutput = programOp;
//     });
//     return success();
//   }
//
// private:
//   calyx::ProgramOp *programOpOutput = nullptr;
//   StringRef topLevelFunction;
// };

struct AddFOpConverter : public OpRewritePattern<arith::AddFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type type = op.getType();
    Value a = op.getLhs();
    Value b = op.getRhs();

    //    Value addi = rewriter.create<arith::AddIOp>(loc, int32Type, a, b);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op, IntegerType::get(type.getContext(), 32, IntegerType::Signless), a,
        b);

    return success();
  }
};

namespace {
class ConvertMulFOp : public OpConversionPattern<arith::MulFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::MulFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    TypeConverter *typeConverter = getTypeConverter();
    auto convertedLhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(a.getType()), a);
    auto convertedRhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(b.getType()), b);
    rewriter.replaceOpWithNewOp<arith::MulIOp>(op, convertedLhs, convertedRhs);

    return success();
  }
};

class ConvertDivFOp : public OpConversionPattern<arith::DivFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::DivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    TypeConverter *typeConverter = getTypeConverter();
    auto convertedLhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(a.getType()), a);
    auto convertedRhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(b.getType()), b);
    rewriter.replaceOpWithNewOp<arith::DivUIOp>(op, convertedLhs, convertedRhs);

    return success();
  }
};

class ConvertAddFOp : public OpConversionPattern<arith::AddFOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::AddFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    TypeConverter *typeConverter = getTypeConverter();
    auto convertedLhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(a.getType()), a);
    auto convertedRhs = typeConverter->materializeTargetConversion(
        rewriter, loc, typeConverter->convertType(b.getType()), b);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, convertedLhs, convertedRhs);

    return success();
  }
};

class ConvertGlobalOp : public OpConversionPattern<memref::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::GlobalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newType = typeConverter->convertType(op.type()).cast<MemRefType>();
    Type elemTy = rewriter.getI32Type();
    auto tensor0D = RankedTensorType::get(op.type().getShape(), elemTy);
    auto init = DenseIntElementsAttr::get(tensor0D, {APInt(32, 1)});

    rewriter.replaceOpWithNewOp<memref::GlobalOp>(
        op, op.sym_name(),
        /*sym_visibility=*/op.sym_visibilityAttr(),
        /*type=*/newType,
        /*initial_value=*/nullptr,
        /*constant=*/true,
        /*alignment=*/nullptr);

    return success();
  }
};

void replaceGlobalMemRefOp(memref::GlobalOp op, TypeConverter &typeConverter,
                           IRRewriter &rewriter) {}

void replaceGetGlobalMemRefOp(memref::GetGlobalOp op,
                              TypeConverter &typeConverter,
                              IRRewriter &rewriter) {
  auto newType = typeConverter.convertType(op.getType()).cast<MemRefType>();
  rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, newType, op.name());
}

} // namespace

namespace {
class HLSQuantize : public HLSQuantizeBase<HLSQuantize> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    auto int32Type = IntegerType::get(context, 32, IntegerType::Signless);
    auto float32Type = Float32Type::get(context);
    target.addLegalOp<UnrealizedConversionCastOp>();
    typeConverter.addConversion(
        [&](Float32Type type) -> Optional<Type> { return int32Type; });
    typeConverter.addTargetMaterialization(
        [&](OpBuilder &builder, Type type, ValueRange inputs,
            Location loc) -> Optional<Value> {
          assert(inputs.size() == 1);
          assert(inputs[0].getType().isa<Float32Type>());
          return builder
              .create<UnrealizedConversionCastOp>(loc, int32Type, inputs[0])
              .getResult(0);
        });
    typeConverter.addSourceMaterialization(
        [&](OpBuilder &builder, Type type, ValueRange inputs,
            Location loc) -> Optional<Value> {
          assert(inputs.size() == 1);
          assert(inputs[0].getType().isa<IntegerType>());
          return builder
              .create<UnrealizedConversionCastOp>(loc, float32Type, inputs[0])
              .getResult(0);
        });

    RewritePatternSet patterns(context);
    target.addLegalOp<arith::AddIOp>();
    target.addIllegalOp<arith::AddFOp>();
    patterns.add<ConvertAddFOp>(typeConverter, context);
    target.addLegalOp<arith::MulIOp>();
    target.addIllegalOp<arith::MulFOp>();
    patterns.add<ConvertMulFOp>(typeConverter, context);
    target.addLegalOp<arith::DivUIOp>();
    target.addIllegalOp<arith::DivFOp>();
    patterns.add<ConvertDivFOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    auto func = getOperation();
    //    func.walk([&](memref::GlobalOp op) { op->dump(); });
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::HLS::createHLSQuantizePass() {
  return std::make_unique<HLSQuantize>();
}

namespace {
class HLSQuantizeModulePass
    : public HLSHLSQuantizeModuleBase<HLSQuantizeModulePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    TypeConverter typeConverter;
    auto int32Type = IntegerType::get(context, 32, IntegerType::Signless);
    typeConverter.addConversion([&](MemRefType type) -> Optional<Type> {
      if (!type.getElementType().isa<Float32Type>())
        return type;

      return MemRefType::get(type.getShape(), int32Type);
    });

    RewritePatternSet patterns(context);
    target.addIllegalOp<memref::GlobalOp>();
    target.addDynamicallyLegalOp<memref::GlobalOp>([&](memref::GlobalOp op) {
      return op.type().getElementType().isa<IntegerType>();
    });
    patterns.add<ConvertGlobalOp>(typeConverter, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();

    auto module = getOperation();
//    OpBuilder builder(module.getBodyRegion());
//    IRRewriter rewriter(builder);
//    module.walk([&](memref::GlobalOp op) {
//      auto newType = typeConverter.convertType(op.type()).cast<MemRefType>();
//      Type elemTy = rewriter.getI32Type();
//      auto tensor0D = RankedTensorType::get(op.type().getShape(), elemTy);
//      auto init = DenseIntElementsAttr::get(tensor0D, {APInt(32, 1)});
//
//      rewriter.replaceOpWithNewOp<memref::GlobalOp>(
//          op, op.sym_name(), op.sym_visibilityAttr(), newType, nullptr, true,
//          nullptr);
//    });
    module->dump();

    module.walk([&](FuncOp func) {
      OpBuilder builder(func.getBody());
      IRRewriter rewriter(builder);
      func.walk([&](memref::GetGlobalOp op) {
        //        IRRewriter rewriter(memref.getContext());
        auto newType =
            typeConverter.convertType(op.getType()).cast<MemRefType>();
//        rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, newType,
//                                                         op.name());
        op.getResult().setType(newType);
//        rewriter.updateRootInPlace(memref,
//                                   [&] { memref.getType().getElementType(); });
      });
      module.dump();
      func.walk([&](memref::LoadOp op) {
        if (op.getMemRef().getType().cast<MemRefType>().getElementType().isa<IntegerType>())
          return WalkResult::advance();

        op->dump();
        std::cout << "\n";
        auto newType =
            typeConverter.convertType(op.getMemRefType());

        op.getResult().setType(newType);
//        newType.dump();
//        std::cout << "\n";
//
//        auto newOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(op, newType, op.getMemRef(), op.indices());
//        newOp->dump();
//        std::cout << "********************\n";
        return WalkResult::advance();
      });
      module.dump();
      func.walk([&](memref::StoreOp op) {
        if (op.getMemRef().getType().cast<MemRefType>().getElementType().isa<IntegerType>())
          return WalkResult::advance();

        auto newType =
            typeConverter.convertType(op.getMemRefType());
        op->getResult(0).setType(newType);
        //        newType.dump();
        //        std::cout << "\n";
        //
        //        auto newOp = rewriter.replaceOpWithNewOp<memref::LoadOp>(op, newType, op.getMemRef(), op.indices());
        //        newOp->dump();
        //        std::cout << "********************\n";
        return WalkResult::advance();
      });
//      func.walk([&](memref::StoreOp memref) {
//        auto newType =
//            typeConverter.convertType(memref.getMemRef().getType()).cast<MemRefType>();
//        rewriter.replaceOpWithNewOp<memref::StoreOp>(memref, newType,
//                                                    memref.indices());
//      });
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::HLS::createHLSHLSQuantizeModulePass() {
  return std::make_unique<HLSQuantizeModulePass>();
}
