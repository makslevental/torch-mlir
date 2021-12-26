//===-------------------------------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HLS_TORCH_TYPES
#define HLS_TORCH_TYPES

include "torch-mlir/Dialect/Torch/IR/TorchBase.td"

def IsOutOfPlaceVariant
  : TorchOpTrait<"IsOutOfPlaceVariant">;

#endif // HLS_TORCH_TYPES
