//===-- TorchBind.td - Torch dialect bind ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"
#include "llvm/ADT/SmallString.h"
#include <mlir/CAPI/IR.h>
#include <mlir/IR/AsmState.h>

namespace py = pybind11;

std::string getValIdent(pybind11::object &pyObjValue) {
  auto _value = mlirPythonCapsuleToValue(pyObjValue.ptr());
  if (mlirValueIsNull(_value))
    return "null";
  llvm::SmallString<32> str;
  llvm::raw_svector_ostream os(str);
  mlir::Value value;
  mlir::Operation *parentOp;
  if (mlirValueIsABlockArgument(_value)) {
    value = unwrap(_value).cast<mlir::BlockArgument>();
    parentOp = value.getParentBlock()->getParentOp();
  } else {
    value = unwrap(_value);
    parentOp = value.getDefiningOp()
                   ->getParentRegion()
                   ->getParentOfType<mlir::func::FuncOp>();
  }
  mlir::AsmState asm_state(parentOp);
  value.printAsOperand(os, asm_state);
  return os.str().str();
}

PYBIND11_MODULE(_torchMlir, m) {
  torchMlirRegisterAllPasses();

  m.doc() = "torch-mlir main python extension";

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__torch__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true);

  m.def("get_val_identifier", &getValIdent);
}
