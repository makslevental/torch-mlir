//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HLS_TORCHTYPES_H
#define HLS_TORCHTYPES_H

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace torch {
namespace Torch {

#define GET_TYPEDEF_CLASSES
#include "BraggHLSTypes.h.inc"

} // namespace HLS
} // namespace torch
} // namespace mlir

#endif // HLS_TORCHTYPES_H
