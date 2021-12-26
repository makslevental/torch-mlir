//===- ivalue_importer.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
#define TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H

#include <memory>

#include "class_annotator.h"
#include "pybind.h"

#include "mlir-c/IR.h"

#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch_mlir {

struct IValueHasher {
  size_t operator()(const c10::IValue &ivalue) const {
    if (ivalue.isObject() || ivalue.isList() || ivalue.isGenericDict()) {
      return std::hash<const void *>()(
          static_cast<const void *>(ivalue.internalToPointer()));
    }

    return c10::IValue::hash(ivalue);
  }
};

struct IValueEq {
  bool operator()(const c10::IValue &lhs, const c10::IValue &rhs) const {
    return lhs.isSameIdentity(rhs);
  }
};

class IValueImporter {
public:
  IValueImporter(MlirBlock importBlock, MlirContext context,
                 ClassAnnotator &annotator)
      : importBlock(importBlock), context(context), annotator(annotator) {}

  MlirValue importIValue(c10::IValue ivalue);

private:
  MlirValue rawImportIValue(c10::IValue ivalue);
  MlirValue importTensor(c10::IValue ivalue);
  MlirValue importModule(torch::jit::Module jitModule);
  void importMethod(torch::jit::Function *function, MlirBlock classTypeBody,
                    const MethodAnnotation &methodAnnotation);
  void importClassType(c10::ClassType *classType);
  void importCompilationUnit(torch::jit::CompilationUnit *cu);

  MlirBlock importBlock;
  MlirContext context;
  ClassAnnotator &annotator;

  // Map tracking already-imported values.
  std::unordered_map<c10::IValue, MlirValue, IValueHasher, IValueEq> valueMap;

  // The unique compilation unit that is shared by all modules reachable
  // from the root ivalue being imported.
  // It basically contains a symbol table of functions which are referenced from
  // e.g. methods (the function names are meaningful and match with Python's
  // module hierarchy, with the exception of `__main__` being replaced with
  // `__torch__`).
  torch::jit::CompilationUnit *compilationUnit = nullptr;

  // Used to detect potentially aliasing tensors.
  std::unordered_set<c10::StorageImpl *> seenStorageImpls;
  // The set of ClassType's that have already been imported.
  //
  // ClassType's are referenced via their `classType->name()->qualifiedName()`
  // string (as an MLIR symbol name) so we don't need to keep a map associating
  // them with the MlirOperation that they import into.
  std::unordered_set<c10::ClassType *> classTypes;
  // The stack of attribute names we have traversed to reach the current IValue.
  // Used for diagnostics.
  std::vector<std::string> attributeNameStack;
  // The root module encountered during recursive IValue traversal.
  // Used for diagnostics.
  // Note that the "top-level" object being imported can in theory be a list
  // of modules, so this is populated when our recursive traversal enters a
  // module not enclosed in any other module, and unset after our recursive
  // traversal exits the module.
  c10::optional<std::string> rootModuleName;
};
/// Main entry-point for importing torch IValue's .
/// Recursively imports `ivalue`, inserting operations at the end of `block`.
void importIValue(c10::IValue ivalue, MlirBlock block, MlirContext context,
                  ClassAnnotator &annotator);

} // namespace torch_mlir

#endif // TORCHMLIRJITIRIMPORTER_CSRC_IVALUE_IMPORTER_H
