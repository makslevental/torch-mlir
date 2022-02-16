# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.
"""Queries the pytorch op registry and generates ODS and CC sources for the ops.
"""

from typing import List, Optional, TextIO

import argparse
import itertools
import logging
import os
import sys

from .utils import TextEmitter
from .registry import Registry, JitOperator

# Mapping from torch types to their corresponding ODS type predicates.
# Use `get_ods_type` instead of using this directly.
TORCH_TYPE_TO_ODS_TYPE = {
    "Tensor": "AnyTorchTensorType",
    "Tensor?": "AnyTorchOptionalTensorType",
    "Tensor?[]": "AnyTorchOptionalTensorListType",
    "Tensor[]": "AnyTorchTensorListType",
    "Scalar": "AnyTorchScalarType",
    "Scalar?": "AnyTorchOptionalScalarType",
    "int": "Torch_IntType",
    "int[]": "TorchIntListType",
    "int?": "TorchOptionalIntType",
    "bool": "Torch_BoolType",
    "bool[]": "TorchBoolListType",
    "bool?": "TorchOptionalBoolType",
    "float": "Torch_FloatType",
    "t[]": "AnyTorchListType",
    "t": "AnyTorchType",
    "t1": "AnyTorchType",
    "t2": "AnyTorchType",
    "Any": "AnyTorchType",
    "Device": "Torch_DeviceType",
    "Device?": "TorchOptionalDeviceType",
    "str": "Torch_StringType",
    "str[]": "TorchStringListType",
    "Dict": "Torch_DictType",
    "__torch__.torch.classes.quantized.LinearPackedParamsBase": "Torch_LinearParamsType",
}


def get_ods_type(type: str):
    # TODO: Increase precision on dict type modeling.
    if type.startswith("Dict("):
      type = "Dict"
    ods_type = TORCH_TYPE_TO_ODS_TYPE.get(type)
    if ods_type is None:
        raise Exception(
            f"{type!r} not in TORCH_TYPE_TO_ODS_TYPE mapping. Please add it!")
    return ods_type


def _get_main_module_name() -> str:
    # pytype: disable=attribute-error
    return sys.modules["__main__"].__loader__.name
    # pytype: enable=attribute-error


ODS_BANNER = "\n".join([
    "//===-------------------------------------------------------*- tablegen -*-===//",
    "//",
    "// This file is licensed under the Apache License v2.0 with LLVM Exceptions.",
    "// See https://llvm.org/LICENSE.txt for license information.",
    "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
    "// Also available under a BSD-style license. See LICENSE.",
    "//",
    "// Operation summaries and descriptions were systematically derived from public",
    "// API docstrings and are licensed accordingly:",
    "//   https://github.com/pytorch/pytorch/blob/master/LICENSE",
    "//===----------------------------------------------------------------------===//",
    "//",
    "// This file is automatically generated.  Please do not edit.",
    "// Generated via:",
    f"//   python -m {_get_main_module_name()}",
    "//",
    "//===----------------------------------------------------------------------===//",
    "",
    "",
])


def raw_emit_op(operator: JitOperator, f: TextIO, *, traits: List[str],
                has_folder: bool, has_canonicalizer: bool):
    """Emit the ODS for a JitOperator to a textual file.

    This is the lowest level of emission and is responsible for low-level
    textual emission details. This function should not have any "smarts"
    for deducing traits/etc.

    You probably don't want to call this directly.
    """
    emitter = TextEmitter(f)
    p = lambda *args: emitter.print(*args)
    op_name, td_def_name = operator.get_mlir_names()

    # Generate unique result names for ops with nameless results
    multiple_results = len(operator.returns) > 1
    generic_result_name = lambda i: "result" + (str(i) if multiple_results else "")

    p(f"def {td_def_name} : Torch_Op<{emitter.quote(op_name)}, [")
    with emitter.indent():
        with emitter.indent():
            p(",\n".join(traits))
        p("]> {")
    with emitter.indent():
        summary = f"Generated op for `{operator.unique_key}`"
        p(f"let summary = {emitter.quote(summary)};")
        p(f"let arguments = (ins")
        with emitter.indent():
            if operator.is_vararg:
                p("Variadic<AnyTorchType>:$operands")
            else:
                p(",\n".join([
                    f"""{get_ods_type(arg["type"])}:${arg["name"]}"""
                    for arg in operator.arguments
                ]))
        p(");")
        p(f"let results = (outs")
        with emitter.indent():
            if operator.is_varret:
                p("Variadic<AnyTorchType>:$results")
            else:
                p(",\n".join([
                    f"""{get_ods_type(ret["type"])}:${ret["name"] or generic_result_name(e)}"""
                    for e, ret in enumerate(operator.returns)
                ]))
        p(");")

        if operator.is_vararg:
            assembly_operands = "`(` $operands `)`"
            assembly_operand_types = "qualified(type($operands))"
        else:
            assembly_operands = " `,` ".join("$" + arg["name"]
                                             for arg in operator.arguments)
            assembly_operand_types = " `,` ".join(
                f"""qualified(type(${arg["name"]}))""" for arg in operator.arguments)
        if operator.is_varret:
            assembly_result_types = "qualified(type($results))"
        else:
            assembly_result_types = " `,` ".join(
                f"""qualified(type(${ret["name"] or generic_result_name(e)}))"""
                for e, ret in enumerate(operator.returns))
        if assembly_operand_types and assembly_result_types:
            maybe_arrow = " `->` "
        else:
            maybe_arrow = ""
        assembly_format = f"{assembly_operands} attr-dict `:` {assembly_operand_types}{maybe_arrow}{assembly_result_types}"
        p(f"let assemblyFormat = {emitter.quote(assembly_format)};")
        if has_folder:
            p("let hasFolder = 1;")
        if has_canonicalizer:
            p("let hasCanonicalizer = 1;")
    p("}")
    p("\n")


def emit_op(operator: JitOperator,
            f: TextIO,
            *,
            traits: Optional[List[str]] = None,
            has_folder: bool = False,
            has_canonicalizer: bool = False):
    """Main entry point for op emission.

    Besides emitting the op, it deduces / adds traits based on the operator
    information.
    """
    if traits is None:
        traits = []

    # All Torch operators allow type refinement.
    traits += ["AllowsTypeRefinement"]
    if operator.has_value_semantics():
        traits += ["HasValueSemantics"]

    raw_emit_op(operator,
                f,
                traits=traits,
                has_folder=has_folder,
                has_canonicalizer=has_canonicalizer)


def emit_prim_ops(torch_ir_dir: str, registry: Registry):
    td_file = os.path.join(torch_ir_dir, "GeneratedPrimOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        emit("prim::layout : (Tensor) -> (int)")
        emit("prim::TupleIndex : (Any, int) -> (Any)", has_canonicalizer=True)
        emit("prim::device : (Tensor) -> (Device)")
        emit("prim::dtype : (Tensor) -> (int)", has_folder=True)
        emit("prim::TupleUnpack : (Any) -> (...)", has_canonicalizer=True)
        emit("prim::NumToTensor.Scalar : (Scalar) -> (Tensor)")
        emit("prim::min.self_int : (int[]) -> (int)")
        emit("prim::min.int : (int, int) -> (int)")
        emit("prim::max.self_int : (int[]) -> (int)")
        emit("prim::max.int : (int, int) -> (int)", has_folder=True)
        emit("prim::RaiseException : (str) -> ()")
        emit("prim::Uninitialized : () -> (Any)", has_canonicalizer=True)
        emit("prim::unchecked_cast : (t) -> (t)",
             traits=["DeclareOpInterfaceMethods<CastOpInterface>"],
             has_folder=True)
        emit("prim::Print : (...) -> ()")
        emit("prim::tolist : (...) -> (...)")


def emit_aten_ops(torch_ir_dir: str, registry: Registry):
    # Note the deliberate lowercasing of the "t" for consistency with all
    # the name munging. This is not load bearing, but is convenient for
    # consistency.
    td_file = os.path.join(torch_ir_dir, "GeneratedAtenOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        def emit_with_mutating_variants(key, **kwargs):
            operator = registry[key]
            emit_op(operator, f, **kwargs)
            ns, unqual, overload = operator.triple
            emit_op(registry.get_by_triple((ns, unqual + "_", overload)),
                    f,
                    traits=["IsTrailingUnderscoreInplaceVariant"])

        # Elementwise tensor compute ops
        for key in [
                "aten::tanh : (Tensor) -> (Tensor)",
                "aten::relu : (Tensor) -> (Tensor)",
                "aten::leaky_relu : (Tensor, Scalar) -> (Tensor)",
                "aten::log : (Tensor) -> (Tensor)",
                "aten::sigmoid : (Tensor) -> (Tensor)",
                "aten::sin : (Tensor) -> (Tensor)",
                "aten::exp : (Tensor) -> (Tensor)",
                "aten::cos : (Tensor) -> (Tensor)",
                "aten::neg : (Tensor) -> (Tensor)",
                "aten::floor : (Tensor) -> (Tensor)",
                "aten::ceil : (Tensor) -> (Tensor)",
                "aten::bitwise_not : (Tensor) -> (Tensor)",
                "aten::add.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::sub.Tensor : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::mul.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::div.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::lerp.Tensor : (Tensor, Tensor, Tensor) -> (Tensor)",
                "aten::eq.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::gt.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::lt.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::ne.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::add.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)",
                "aten::sub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)",
                "aten::mul.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::div.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::ne.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::eq.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::gt.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::ge.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::lt.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::fmod.Scalar : (Tensor, Scalar) -> (Tensor)",
                "aten::masked_fill.Scalar : (Tensor, Tensor, Scalar) -> (Tensor)",
                "aten::clamp : (Tensor, Scalar?, Scalar?) -> (Tensor)",
                "aten::log2 : (Tensor) -> (Tensor)",
                "aten::rsqrt : (Tensor) -> (Tensor)",
                "aten::abs : (Tensor) -> (Tensor)",
                "aten::reciprocal : (Tensor) -> (Tensor)",
                "aten::bitwise_and.Tensor : (Tensor, Tensor) -> (Tensor)",
                "aten::threshold : (Tensor, Scalar, Scalar) -> (Tensor)",

        ]:
            emit_with_mutating_variants(key)
        # Elementwise tensor compute ops that don't have the standard mutating
        # variants.
        emit("aten::addcmul : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)")
        emit("aten::addcdiv : (Tensor, Tensor, Tensor, Scalar) -> (Tensor)")
        emit("aten::maximum : (Tensor, Tensor) -> (Tensor)")
        emit("aten::minimum : (Tensor, Tensor) -> (Tensor)")
        emit("aten::rsub.Scalar : (Tensor, Scalar, Scalar) -> (Tensor)")
        emit("aten::gelu : (Tensor) -> (Tensor)")
        emit("aten::pow.Tensor_Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::threshold_backward : (Tensor, Tensor, Scalar) -> (Tensor)")

        emit_with_mutating_variants("aten::triu : (Tensor, int) -> (Tensor)")
        emit_with_mutating_variants("aten::index_put : (Tensor, Tensor?[], Tensor, bool) -> (Tensor)")

        # Non-elementwise tensor compute ops
        emit("aten::linear : (Tensor, Tensor, Tensor?) -> (Tensor)")
        emit("aten::mm : (Tensor, Tensor) -> (Tensor)")
        emit("aten::addmm : (Tensor, Tensor, Tensor, Scalar, Scalar) -> (Tensor)")
        emit("aten::matmul : (Tensor, Tensor) -> (Tensor)")
        emit(
            "aten::conv2d : (Tensor, Tensor, Tensor?, int[], int[], int[], int) -> (Tensor)"
        )
        emit(
            "aten::batch_norm : (Tensor, Tensor?, Tensor?, Tensor?, Tensor?, bool, float, float, bool) -> (Tensor)"
        )
        emit(
            "aten::layer_norm : (Tensor, int[], Tensor?, Tensor?, float, bool) -> (Tensor)"
        )
        emit (
            "aten::native_layer_norm : (Tensor, int[], Tensor?, Tensor?, float) -> (Tensor, Tensor, Tensor)"
        )
        emit(
            "aten::max_pool2d : (Tensor, int[], int[], int[], int[], bool) -> (Tensor)"
        )
        emit(
            "aten::softmax.int : (Tensor, int, int?) -> (Tensor)"
        )
        emit(
            "aten::log_softmax.int : (Tensor, int, int?) -> (Tensor)"
        )
        emit("aten::adaptive_avg_pool2d : (Tensor, int[]) -> (Tensor)")
        emit("aten::topk : (Tensor, int, int, bool, bool) -> (Tensor, Tensor)")
        emit("aten::transpose.int : (Tensor, int, int) -> (Tensor)")
        emit("aten::permute : (Tensor, int[]) -> (Tensor)")
        emit("aten::bmm : (Tensor, Tensor) -> (Tensor)")
        emit("aten::cumsum : (Tensor, int, int?) -> (Tensor)")
        emit("aten::floor_divide.Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::logsumexp : (Tensor, int[], bool) -> (Tensor)")
        emit("aten::mean.dim : (Tensor, int[], bool, int?) -> (Tensor)")
        emit("aten::__and__.Tensor : (Tensor, Tensor) -> (Tensor)")
        emit("aten::sqrt : (Tensor) -> (Tensor)")
        emit("aten::_softmax : (Tensor, int, bool) -> (Tensor)")
        emit("aten::mean : (Tensor, int?) -> (Tensor)")
        emit("aten::nll_loss_forward : (Tensor, Tensor, Tensor?, int, int) -> (Tensor, Tensor)")

        # Misc tensor ops.
        emit("aten::constant_pad_nd : (Tensor, int[], Scalar) -> (Tensor)")
        emit("aten::squeeze.dim : (Tensor, int) -> (Tensor)", has_folder=True)
        emit("aten::unsqueeze : (Tensor, int) -> (Tensor)")
        emit("aten::squeeze : (Tensor) -> (Tensor)", has_folder=True)
        emit("aten::flatten.using_ints : (Tensor, int, int) -> (Tensor)")
        emit("aten::dim : (Tensor) -> (int)", has_folder=True)
        emit("aten::size : (Tensor) -> (int[])", has_canonicalizer=True)
        emit("aten::fill_.Scalar : (Tensor, Scalar) -> (Tensor)")
        emit("aten::Bool.Tensor : (Tensor) -> (bool)")
        emit("aten::ones : (int[], int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::zeros : (int[], int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::tensor : (t[], int?, Device?, bool) -> (Tensor)")
        emit("aten::tensor.bool : (bool, int?, Device?, bool) -> (Tensor)")
        emit("aten::tensor.int : (int, int?, Device?, bool) -> (Tensor)")
        emit("aten::_shape_as_tensor : (Tensor) -> (Tensor)")
        emit("aten::all : (Tensor) -> (Tensor)")
        emit("aten::any : (Tensor) -> (Tensor)")
        emit("aten::any.dim : (Tensor, int, bool) -> (Tensor)")
        emit("aten::arange : (Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::arange.start : (Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::arange.start_step : (Scalar, Scalar, Scalar, int?, int?, Device?, bool?) -> (Tensor)")
        emit("aten::argmax : (Tensor, int?, bool) -> (Tensor)")
        emit("aten::bucketize.Tensor : (Tensor, Tensor, bool, bool) -> (Tensor)")
        emit("aten::contiguous : (Tensor, int) -> (Tensor)")
        emit("aten::copy_ : (Tensor, Tensor, bool) -> (Tensor)")
        emit("aten::detach : (Tensor) -> (Tensor)")
        emit("aten::embedding : (Tensor, Tensor, int, bool, bool) -> (Tensor)")
        emit("aten::empty_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::zeros_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::ones_like : (Tensor, int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::empty.memory_format : (int[], int?, int?, Device?, bool?, int?) -> (Tensor)")
        emit("aten::expand : (Tensor, int[], bool) -> (Tensor)")
        emit("aten::broadcast_to : (Tensor, int[]) -> (Tensor)")
        emit("aten::index.Tensor : (Tensor, Tensor?[]) -> (Tensor)")
        emit("aten::index_select : (Tensor, int, Tensor) -> (Tensor)")
        emit("aten::item : (Tensor) -> (Scalar)")
        emit("aten::masked_select : (Tensor, Tensor) -> (Tensor)")
        emit("aten::numel : (Tensor) -> (int)")
        emit("aten::repeat : (Tensor, int[]) -> (Tensor)")
        emit("aten::reshape : (Tensor, int[]) -> (Tensor)")
        emit("aten::resize_ : (Tensor, int[], int?) -> (Tensor)")
        emit("aten::select.int : (Tensor, int, int) -> (Tensor)")
        emit("aten::size.int : (Tensor, int) -> (int)", has_folder=True)
        emit("aten::stack : (Tensor[], int) -> (Tensor)")
        emit("aten::sum : (Tensor, int?) -> (Tensor)")
        emit("aten::sum.dim_IntList : (Tensor, int[], bool, int?) -> (Tensor)")
        emit("aten::to.dtype : (Tensor, int, bool, bool, int?) -> (Tensor)", has_folder=True)
        emit("aten::to.other : (Tensor, Tensor, bool, bool, int?) -> (Tensor)")
        emit("aten::to.prim_Device : (Tensor, Device?, int?, bool, bool) -> (Tensor)")
        emit("aten::type_as : (Tensor, Tensor) -> (Tensor)")
        emit("aten::view : (Tensor, int[]) -> (Tensor)", has_folder=True)
        emit("aten::where.self : (Tensor, Tensor, Tensor) -> (Tensor)")
        emit("aten::slice.Tensor : (Tensor, int, int?, int?, int) -> (Tensor)")
        emit("aten::len.Tensor : (Tensor) -> (int)")
        emit("aten::cpu : (Tensor) -> (Tensor)")
        emit("aten::gather : (Tensor, int, Tensor, bool) -> (Tensor)")
        emit("aten::IntImplicit : (Tensor) -> (int)")
        emit("aten::tensor.float : (float, int?, Device?, bool) -> (Tensor)")
        emit("aten::Int.Tensor : (Tensor) -> (int)", has_folder=True)
        emit("aten::dropout : (Tensor, float, bool) -> (Tensor)")
        emit("aten::t : (Tensor) -> (Tensor)")

        # Dict ops.
        emit("aten::__contains__.str : (Dict(str, t), str) -> (bool)", has_folder=True)
        emit("aten::__getitem__.Dict_str : (Dict(str, t), str) -> (t)", has_folder=True)
        emit("aten::_set_item.str : (Dict(str, t), str, t) -> ()")
        emit("aten::keys.str : (Dict(str, t)) -> (str[])")
        emit("aten::get.default_str : (Dict(str, t), str, t) -> (t)")
        emit("aten::Delete.Dict_str : (Dict(str, t), str) -> ()")

        # List ops.
        emit("aten::cat : (Tensor[], int) -> (Tensor)")
        emit("aten::append.t : (t[], t) -> (t[])")
        emit("aten::add.t : (t[], t[]) -> (t[])")
        emit("aten::eq.int_list : (int[], int[]) -> (bool)", has_folder=True)
        emit("aten::list.t : (t[]) -> (t[])")
        emit("aten::slice.t : (t[], int?, int?, int) -> (t[])")
        emit("aten::insert.t : (t[], int, t) -> ()")

        # Str ops.
        emit("aten::add.str : (str, str) -> (str)")
        emit("aten::eq.str : (str, str) -> (bool)", has_folder=True)
        emit("aten::str : (t) -> (str)")
        emit("aten::format : (...) -> (str)")
        emit("aten::join : (str, str[]) -> (str)")

        # Type conversion ops.
        emit("aten::Float.Scalar : (Scalar) -> (float)", has_folder=True)
        emit("aten::Float.str : (str) -> (float)")
        emit("aten::Int.float : (float) -> (int)")

        # Primitive ops
        emit("aten::__range_length : (int, int, int) -> (int)", has_folder=True)
        emit("aten::__derive_index : (int, int, int) -> (int)", has_folder=True)
        emit("aten::gt.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::ge.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::lt.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::le.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::ne.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::eq.int : (int, int) -> (bool)", has_folder=True)
        emit("aten::floordiv.int : (int, int) -> (int)", has_folder=True)
        emit("aten::remainder.int : (int, int) -> (int)", has_folder=True)
        emit("aten::add.int : (int, int) -> (int)", has_folder=True)
        emit("aten::sub.int : (int, int) -> (int)", has_folder=True)
        emit("aten::mul.int : (int, int) -> (int)", has_folder=True)
        emit("aten::neg.int : (int) -> (int)", has_folder=True)
        emit("aten::log.int : (int) -> (float)")
        emit("aten::add.float_int : (float, int) -> (float)")
        emit("aten::mul.float : (float, float) -> (float)")
        emit("aten::neg.float : (float) -> (float)")
        emit("aten::lt.float_int : (float, int) -> (bool)")
        emit("aten::eq.float : (float, float) -> (bool)", has_folder=True)
        emit("aten::__and__.bool : (bool, bool) -> (bool)")
        emit("aten::ne.bool : (bool, bool) -> (bool)", has_folder=True)
        emit("aten::__is__ : (t1, t2) -> (bool)", has_folder=True)
        emit("aten::__isnot__ : (t1, t2) -> (bool)", has_folder=True)
        emit("aten::__not__ : (bool) -> (bool)", has_folder=True)
        emit("aten::len.t : (t[]) -> (int)",
             has_folder=True,
             has_canonicalizer=True)
        emit("aten::__getitem__.t : (t[], int) -> (t)", has_canonicalizer=True)
        emit("aten::_set_item.t : (t[], int, t) -> (t[])")
        emit("aten::div : (Scalar, Scalar) -> (float)")
        emit("aten::eq.device : (Device, Device) -> (bool)")

        # backprop ops
        emit("aten::_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)")
        emit("aten::tanh_backward : (Tensor, Tensor) -> (Tensor)")
        emit("aten::gelu_backward : (Tensor, Tensor) -> (Tensor)")
        emit("aten::_log_softmax_backward_data : (Tensor, Tensor, int, int) -> (Tensor)")



def emit_quantized_ops(torch_ir_dir: str, registry: Registry):
    td_file = os.path.join(torch_ir_dir, "GeneratedQuantizedOps.td")
    with open(td_file, "w") as f:
        f.write(ODS_BANNER)

        def emit(key, **kwargs):
            emit_op(registry[key], f, **kwargs)

        emit(
            "quantized::linear : (Tensor, __torch__.torch.classes.quantized.LinearPackedParamsBase, float, int) -> (Tensor)",
            traits=["HasValueSemantics"])


def dump_registered_ops(outfile: TextIO, registry: Registry):
    for _, v in sorted(registry.by_unique_key.items()):
        outfile.write(repr(v))

def main(args: argparse.Namespace):
    registry = Registry.load()
    if args.debug_registry_dump:
        with open(args.debug_registry_dump, "w") as debug_registry_dump:
            dump_registered_ops(debug_registry_dump, registry)
    emit_prim_ops(args.torch_ir_dir, registry)
    emit_aten_ops(args.torch_ir_dir, registry)
    emit_quantized_ops(args.torch_ir_dir, registry)


def _create_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_ods")
    parser.add_argument(
        "--torch_ir_dir",
        required=True,
        help="Directory containing the Torch dialect definition")
    parser.add_argument(
        "--debug_registry_dump",
        help="File to dump the the PyTorch JIT operator registry into")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = _create_argparse()
    args = parser.parse_args()
    main(args)
