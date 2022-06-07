//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HLS_PASSES_H
#define HLS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace BraggHLS {

std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSRefineTypesPass();

std::unique_ptr<OperationPass<ModuleOp>> createBraggHLSDropPublicReturnPass();

std::unique_ptr<OperationPass<ModuleOp>> createBraggHLSPromoteAllocsPass();

std::unique_ptr<OperationPass<ModuleOp>> createBraggHLSHLSQuantizeModulePass();

std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSLinalgBufferizePass();

std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSDecomposeComplexOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSConvertTorchToLinalgPass();

std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSQuantizePass();

std::unique_ptr<OperationPass<ModuleOp>> createBraggHLSHLSQuantizeModulePass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertCopyToAffineLoopsPass();

//std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyMemrefAccessPass();

// std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSReduceOpVariantsPass();

// std::unique_ptr<OperationPass<func::FuncOp>> createBraggHLSConvertOperatorsPass();

// std::unique_ptr<OperationPass<ModuleOp>>
// createBraggHLSAdjustCallingConventionsPass();

void createTorchBackendToLinalgOnTensorsBackendPipeline(OpPassManager &pm);

void createTorchScriptModuleToTorchHLSBackendPipeline(OpPassManager &pm);

void createTorchScriptFunctionToTorchHLSBackendPipeline(OpPassManager &pm);

void registerBraggHLSPasses();

void registerHLSConversionPasses();

void registerHLSPasses();

void registerDirtyPass();

} // namespace HLS
} // namespace mlir

#endif // HLS_PASSES_H
