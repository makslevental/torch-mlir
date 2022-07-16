#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallString.h"
#include <mlir/CAPI/IR.h>
#include <mlir/IR/AsmState.h>

namespace py = pybind11;

#define MLIR_PYTHON_CAPSULE_OP_RESULT MAKE_MLIR_PYTHON_QUALNAME("ir.OpResult._CAPIPtr")

static inline MlirValue mlirPythonCapsuleToOpResult(PyObject *capsule) {
  void *ptr = PyCapsule_GetPointer(capsule, MLIR_PYTHON_CAPSULE_OP_RESULT);
  MlirValue value = {ptr};
  return value;
}

std::string getValIdent(py::object pyObjValue) {
  auto _value = mlirPythonCapsuleToOpResult(pyObjValue.ptr());
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
