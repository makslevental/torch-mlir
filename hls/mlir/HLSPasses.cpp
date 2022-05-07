#include "HLSPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToStd/TorchToStd.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "llvm/PassInfo.h"

using namespace mlir;
using namespace mlir::torch::Torch;
using namespace mlir::torch::HLS;

namespace {
#define GEN_PASS_REGISTRATION
#include "HLSPasses.h.inc"
} // end namespace

void mlir::torch::HLS::registerHLSPasses() {
  ::registerPasses();
  registerHLSTorchPasses();
  registerHLSConversionPasses();
}

using namespace mlir::torch;
using namespace mlir::torch::HLS;
using namespace mlir::torch::TorchConversion;

void mlir::torch::HLS::registerHLSTorchPasses() {
  ::registerPasses();
  mlir::PassPipelineRegistration<>(
      "torchscript-module-to-torch-hls-backend-pipeline", "",
      mlir::torch::HLS::createTorchScriptModuleToTorchHLSBackendPipeline);
  mlir::PassPipelineRegistration<>(
      "torch-function-to-torch-hls-backend-pipeline", "",
      mlir::torch::HLS::createTorchScriptFunctionToTorchHLSBackendPipeline);

//  registerDirtyPass();
}

void mlir::torch::HLS::createTorchScriptModuleToTorchHLSBackendPipeline(
    OpPassManager &pm) {
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createPrepareForGlobalizeObjectGraphPass());
  pm.addPass(createGlobalizeObjectGraphPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createInlinerPass());

  createTorchScriptFunctionToTorchHLSBackendPipeline(pm);
}

namespace mlir {
namespace torch {
namespace HLS {

void createTorchScriptFunctionToTorchHLSBackendPipeline(OpPassManager &pm) {
  pm.addPass(createAdjustCallingConventionsPass());

  if (true) {
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createInlineGlobalSlotsPass());
  }

  pm.addNestedPass<func::FuncOp>(createReduceOpVariantsPass());

  if (true) {
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addPass(createSymbolDCEPass());
  }

  // ours
  pm.addNestedPass<func::FuncOp>(HLS::createHLSRefineTypesPass());

  pm.addPass(Torch::createRefinePublicReturnPass());

  if (true) {
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  pm.addNestedPass<func::FuncOp>(Torch::createMaximizeValueSemanticsPass());

  if (true) {
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  //  pm.addPass(HLS::createHLSDropPublicReturnPass());
  pm.addNestedPass<func::FuncOp>(Torch::createDecomposeComplexOpsPass());
  pm.addNestedPass<func::FuncOp>(HLS::createHLSDecomposeComplexOpsPass());

  // TODO: VerifyHLSTorchBackendContractPass.
}

void registerHLSConversionPasses() {
  ::registerPasses();
  ::mlir::PassPipelineRegistration<>(
      "torch-hls-backend-to-linalg-on-tensors-backend-pipeline", "",
      createTorchBackendToLinalgOnTensorsBackendPipeline);
}

void createTorchBackendToLinalgOnTensorsBackendPipeline(OpPassManager &pm) {
  //  pm.addNestedPass<func::FuncOp>(createHLSConvertOperatorsPass());
  // Check some invariants to catch errors in a clear way.
  pm.addPass(
      TorchConversion::createVerifyInvariantsBeforeBackendLoweringPass());

  // Lower to linalg + guards which is the input to codegen backends.
  // We do this first as it tends to involve pattern-matching against constants,
  // (e.g. dimensions which must be constant in a ranked programming model)
  // and those constants get somewhat obscured by TorchToStd.
  pm.addNestedPass<func::FuncOp>(createHLSConvertTorchToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToLinalgPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToStdPass());
  pm.addNestedPass<func::FuncOp>(createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());

  if (true) {
    // Clean up any non-canonical code introduced above..
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    // Resolve `dim` ops on tensors (which currently live in the `memref`
    // dialect for some reason -- we don't have memrefs at this level).
    pm.addNestedPass<func::FuncOp>(memref::createResolveShapedTypeResultDimsPass());
    // The resolution of `dim` ops tends to create identical ops. CSE them.
    pm.addNestedPass<func::FuncOp>(createCSEPass());
  }

  // Finish the type conversion from `torch` types to the types of the
  // linalg-on-tensors backend contract.
  pm.addPass(TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(
      TorchConversion::createFinalizingBackendTypeConversionPass());

  // Verify that we have lowered to the form that linalg on tensors backends
  // expect. This fails compilation (signalPassFailure) if the IR is not in the
  // correct form.
  pm.addPass(TorchConversion::createVerifyLinalgOnTensorsBackendContractPass());
}
} // namespace HLS
} // namespace torch
} // namespace mlir