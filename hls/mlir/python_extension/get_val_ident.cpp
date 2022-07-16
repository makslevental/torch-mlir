#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallString.h"
#include <mlir/CAPI/IR.h>
#include <mlir/IR/AsmState.h>

namespace py = pybind11;

std::string getValIdent(py::object pyObjValue) {
  auto _value = mlirPythonCapsuleToValue(pyObjValue.ptr());
  if (mlirValueIsNull(_value))
    return "null";
  mlirValueDump(_value);
  auto value = unwrap(_value);
  llvm::SmallString<32> str;
  llvm::raw_svector_ostream os(str);
  mlir::AsmState asm_state(
      value.getDefiningOp()->getParentRegion()->getParentOfType<mlir::func::FuncOp>());
  value.printAsOperand(os, asm_state);
  return os.str().str();
}

PYBIND11_MODULE(hls_utils, m) {
  m.def("get_val_identifier", &getValIdent);
}
