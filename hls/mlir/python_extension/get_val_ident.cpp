#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "torch-mlir-c/Dialects.h"
#include "torch-mlir-c/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallString.h"
#include <mlir/CAPI/IR.h>
#include <mlir/IR/AsmState.h>


namespace py = pybind11;

std::string getValIdent(pybind11::object& pyObjValue) {
  auto _value = mlirPythonCapsuleToValue(pyObjValue.ptr());
  if (mlirValueIsNull(_value))
    return "null";
  mlirValueDump(_value);
  auto value = unwrap(_value);
  llvm::SmallString<32> str;
  llvm::raw_svector_ostream os(str);
  mlir::AsmState asm_state(value.getDefiningOp()
                               ->getParentRegion()
                               ->getParentOfType<mlir::func::FuncOp>());
  value.printAsOperand(os, asm_state);
  return os.str().str();
}

PYBIND11_MODULE(hls_utils, m) { m.def("get_val_identifier", &getValIdent); }
