#include "HLSPassDetail.h"
#include "HLSPasses.h"
#include "BraggHLSOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include <iostream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::BraggHLS;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

namespace {
class ConvertOperatorOp : public OpConversionPattern<Torch::OperatorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    op->dump();
    auto operands = op.operands();
    auto name = op->getAttrDictionary().getNamed("name");

    if (name.hasValue()) {
      auto name_ = name->getValue().cast<StringAttr>().str();
      std::cout << name_ << std::endl;
      if (name_ == "aten.mm.out") {
        assert(operands.size() == 3);
        rewriter.replaceOpWithNewOp<AtenMmOp>(op, op.getType(0), operands[0],
                                                 operands[1]);
      }
      op->dump();
      return success();
    } else {
      return failure();
    }
  }
};
} // namespace

namespace {

class HLSConvertOperators
    : public mlir::BraggHLS::HLSConvertOperatorsBase<HLSConvertOperators> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<StandardOpsDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<arith::ArithmeticDialect>();
    registry.insert<TorchDialect>();
    TorchConversion::getBackendTypeConversionDependentDialects(registry);
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect,
                           math::MathDialect, tensor::TensorDialect,
                           arith::ArithmeticDialect, Torch::TorchDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    TorchConversion::setupBackendTypeConversion(target, typeConverter);

    RewritePatternSet patterns(context);

    target.addIllegalOp<OperatorOp>();
    patterns.add<ConvertOperatorOp>(typeConverter, context);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::BraggHLS::createBraggHLSConvertOperatorsPass() {
  return std::make_unique<HLSConvertOperators>();
}
