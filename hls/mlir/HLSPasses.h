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
namespace torch {
namespace HLS {

std::unique_ptr<OperationPass<func::FuncOp>> createHLSRefineTypesPass();

std::unique_ptr<OperationPass<ModuleOp>> createHLSDropPublicReturnPass();

std::unique_ptr<OperationPass<ModuleOp>> createHLSPromoteAllocsPass();

std::unique_ptr<OperationPass<ModuleOp>> createHLSHLSQuantizeModulePass();

std::unique_ptr<OperationPass<func::FuncOp>> createHLSLinalgBufferizePass();

std::unique_ptr<OperationPass<func::FuncOp>> createHLSDecomposeComplexOpsPass();

std::unique_ptr<OperationPass<func::FuncOp>> createHLSConvertTorchToLinalgPass();

std::unique_ptr<OperationPass<func::FuncOp>> createHLSQuantizePass();

std::unique_ptr<OperationPass<ModuleOp>> createHLSHLSQuantizeModulePass();

std::unique_ptr<OperationPass<func::FuncOp>> createConvertCopyToAffineLoopsPass();

//std::unique_ptr<OperationPass<func::FuncOp>> createSimplifyMemrefAccessPass();

// std::unique_ptr<OperationPass<func::FuncOp>> createHLSReduceOpVariantsPass();

// std::unique_ptr<OperationPass<func::FuncOp>> createHLSConvertOperatorsPass();

// std::unique_ptr<OperationPass<ModuleOp>>
// createHLSAdjustCallingConventionsPass();

void createTorchBackendToLinalgOnTensorsBackendPipeline(OpPassManager &pm);

void createTorchScriptModuleToTorchHLSBackendPipeline(OpPassManager &pm);

void createTorchScriptFunctionToTorchHLSBackendPipeline(OpPassManager &pm);

void registerHLSTorchPasses();

void registerHLSConversionPasses();

void registerHLSPasses();

void registerDirtyPass();

} // namespace HLS
} // namespace torch
} // namespace mlir

#endif // HLS_PASSES_H
