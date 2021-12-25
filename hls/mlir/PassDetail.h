//===- PassDetail.h - Pass details ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef HLS_PASSDETAIL_H
#define HLS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace torch {
namespace HLS {

#define GEN_PASS_CLASSES
#include "Passes.h.inc"

} // namespace HLS
} // namespace torch
} // end namespace mlir

#endif // HLS_PASSDETAIL_H
