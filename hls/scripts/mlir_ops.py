from __future__ import annotations

import _ast
import argparse
import ast
import inspect
from _ast import Subscript
from ast import Assign, Mult, Add, BinOp, Name, Call, keyword, Str, IfExp, Compare
from typing import Tuple, Union, Dict, Any

import astor
import numpy as np

MAC_IDX = None

OUTPUT_ARRAYS = []

from dataclasses import dataclass, field

ArrayIndex = Tuple[int]

PEIndex = Tuple[int]

DTYPE = "half"


@dataclass(frozen=True)
class _Instruction:
    pe_id: PEIndex = field(init=False, default_factory=lambda: CURRENT_PE)


@dataclass(frozen=True)
class NOP(_Instruction):
    pass


@dataclass(frozen=True)
class Bin(_Instruction):
    left: Any
    right: Any


@dataclass(frozen=True)
class MulInst(Bin):
    pass


@dataclass(frozen=True)
class DivInst(Bin):
    pass


@dataclass(frozen=True)
class AddInst(Bin):
    pass


@dataclass(frozen=True)
class SubInst(Bin):
    pass


@dataclass(frozen=True)
class GTInst(Bin):
    pass


@dataclass(frozen=True)
class ReLUInst(_Instruction):
    val: Any


@dataclass(frozen=True)
class ExpInst(_Instruction):
    val: Any


Instruction = Union[NOP, ExpInst, ReLUInst, MulInst, AddInst, DivInst, SubInst, GTInst]

# def assert_never(x: NoReturn) -> NoReturn:
#     raise AssertionError("Unhandled type: {}".format(type(x).__name__))
#
# def showResult(r: Result) -> str:
#     if isinstance(r, OK):
#         return str(r.result)
#     elif isinstance(r, Failure):
#         return "Failure: " + r.msg
#     else:
#         assert_never(r)

CURRENT_PE: PEIndex = None


class PE:
    def __init__(self, index):
        self.index = index
        self._instructions = []

    def push_instructions(self, inst):
        self._instructions.append(inst)

    def num_instructions(self):
        return len(self._instructions)

    def push_nop(self):
        self._instructions.append(NOP())


PES: Dict[PEIndex, PE] = {}


def index_map(index, curr_shape, prev_shape):
    return tuple(np.unravel_index(np.ravel_multi_index(index, curr_shape), prev_shape))


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_array_type(shape, ptr=True, nd=False):
    if nd:
        typ = ""
        for s in shape:
            typ += f"[{s} x "
        typ += DTYPE
        for _ in shape:
            typ += "]"
        if ptr:
            typ += "*"
    else:
        typ = f"{np.prod(shape)} x {DTYPE}"
    return typ


class RemoveMulAdd(ast.NodeTransformer):
    subs = {}
    dels = set()

    def visit_For(self, node):
        self.generic_visit(node)
        assigns = [b for b in node.body if isinstance(b, Assign)]
        if len(assigns) > 1:
            for i in range(len(assigns) - 1):
                assign1, assign2 = assigns[i], assigns[i + 1]
                if (
                    isinstance(assign1.value, BinOp)
                    and isinstance(assign1.value.op, Mult)
                    and isinstance(assign2.value, BinOp)
                    and isinstance(assign2.value.op, Add)
                ):
                    self.dels.add(assign1)
                    self.subs[assign2] = (
                        assign1.value.left,
                        assign1.value.right,
                        assign2.value.left,
                    )
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.dels:
            return None
        elif node in self.subs:
            return Assign(
                targets=node.targets,
                value=Call(func=Name(id="FMulAdd"), args=self.subs[node], keywords=[]),
                type_comment=None,
            )
        else:
            return node

    def visit_AugAssign(self, node):
        return node


class RemoveMulAddAndSubscript(ast.NodeTransformer):
    subs = {}
    dels = set()

    def visit_For(self, node):
        self.generic_visit(node)
        assigns = [b for b in node.body if isinstance(b, Assign)]
        for i in range(len(assigns) - 2):
            assign1, assign2, assign3 = assigns[i : i + 3]
            if (
                isinstance(assign1.value, Subscript)
                and (
                    (
                        isinstance(assign2.value, Call)
                        and assign2.value.func.id == "FMulAdd"
                    )
                    or (
                        isinstance(assign2.value, BinOp)
                        and isinstance(assign2.value.op, Add)
                    )
                )
                and isinstance(assign3.targets[0], Subscript)
            ):
                self.dels.add(assign1)
                self.dels.add(assign3)
                keywords = [
                    keyword(arg="arr", value=assign3.targets[0].value),
                    keyword(
                        arg="idx",
                        value=_ast.Tuple((tuple(e for e in assign1.value.slice.elts))),
                    ),
                ]
                if isinstance(assign2.value, BinOp):
                    assign2.value = Call(
                        func=Name(id="Add"),
                        args=[assign2.value.left],
                        keywords=[],
                    )
                else:
                    assign2.value = Call(
                        func=Name(id="FMac"),
                        args=assign2.value.args[:-1],
                        keywords=[],
                    )
                assign2.value.keywords = keywords
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.dels:
            return None
        else:
            return node

    def visit_AugAssign(self, node):
        return node


class RemoveMAC(ast.NodeTransformer):
    subs = {}
    dels = set()
    body_args = []

    def visit_FunctionDef(self, node):
        if node.name == "body":
            self.body_args = node.args.args
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        assigns = [b for b in node.body if isinstance(b, Assign)]
        if len(assigns) > 1:
            for i in range(len(assigns) - 1):
                assign1, assign2 = assigns[i], assigns[i + 1]
                if (
                    isinstance(assign1.value, BinOp)
                    and isinstance(assign1.value.op, Mult)
                    and isinstance(assign2.value, BinOp)
                    and isinstance(assign2.value.op, Add)
                ):
                    self.dels.add(assign1)
                    self.subs[assign2] = (
                        assign1.value.left,
                        assign1.value.right,
                        assign2.value.left,
                    )
        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.dels:
            return None
        elif node in self.subs:
            mac_inst = Call(func=Name(id="MAC"), args=self.body_args, keywords=[])
            return Assign(
                targets=node.targets,
                value=Call(
                    func=mac_inst,
                    args=self.subs[node],
                    keywords=[keyword(arg="type", value=Str("MulAdd"))],
                ),
                type_comment=None,
            )
        else:
            return node

    def visit_AugAssign(self, node):
        return node


class RemoveMulOrAdd(ast.NodeTransformer):
    subs = {}
    body_args = []

    def visit_FunctionDef(self, node):
        if node.name == "body":
            self.body_args = node.args.args
            assigns = [b for b in node.body if isinstance(b, Assign)]
            for assign in assigns:
                if (
                    isinstance(assign.value, BinOp)
                    and isinstance(assign.value.op, Mult)
                ) or (
                    isinstance(assign.value, BinOp) and isinstance(assign.value.op, Add)
                ):
                    self.subs[assign] = (assign.value.left, assign.value.right)

        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.subs:
            mac_inst = Call(func=Name(id="MAC"), args=self.body_args, keywords=[])
            return Assign(
                targets=node.targets,
                value=Call(
                    func=mac_inst,
                    args=self.subs[node],
                    keywords=[keyword(arg="type", value=Str(node.value.op.__doc__))],
                ),
                type_comment=None,
            )
        else:
            return node

    def visit_AugAssign(self, node):
        return node


class RemoveMul(ast.NodeTransformer):
    subs = {}
    dels = set()
    body_args = []

    def visit_FunctionDef(self, node):
        if node.name == "body":
            self.body_args = node.args.args
        self.generic_visit(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        assigns = [b for b in node.body if isinstance(b, Assign)]
        for assign in assigns:
            if isinstance(assign.value, BinOp) and isinstance(assign.value.op, Mult):
                self.subs[assign] = (assign.value.left, assign.value.right)

        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.subs:
            mac_inst = Call(func=Name(id="Mul"), args=self.body_args, keywords=[])
            return Assign(
                targets=node.targets,
                value=Call(func=mac_inst, args=self.subs[node], keywords=[]),
                type_comment=None,
            )
        else:
            return node


class RemoveIfExp(ast.NodeTransformer):
    subs = {}
    dels = set()
    body_args = []

    def visit_FunctionDef(self, node):
        if node.name == "body":
            self.body_args = node.args.args
            assigns = [b for b in node.body if isinstance(b, Assign)]
            if len(assigns) > 1:
                for i in range(len(assigns) - 1):
                    assign1, assign2 = assigns[i], assigns[i + 1]
                    if isinstance(assign1.value, Compare) and isinstance(
                        assign2.value, IfExp
                    ):
                        self.dels.add(assign1)
                        self.subs[assign2] = (assign1.value.left,)

        self.generic_visit(node)
        return node

    def visit_Assign(self, node):
        if node in self.dels:
            return None
        elif node in self.subs:
            return Assign(
                targets=node.targets,
                value=Call(func=Name(id="ReLU"), args=self.subs[node], keywords=[]),
                type_comment=None,
            )
        else:
            return node

    def visit_AugAssign(self, node):
        return node


class HoistGlobals(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "forward":
            assigns = [(i, b) for i, b in enumerate(node.body) if isinstance(b, Assign) and isinstance(b.value, Call) and isinstance(b.value.func, Name) and b.value.func.id == 'GlobalArray']
            for i, a in reversed(assigns):
                del node.body[i]
                node.args.args.append(a.targets[0].id)
                node.args.defaults.append(a.value)
        return node


# def unroll_loops():
#     passer = apply_passes([loop_unroll()], env=SymbolTable(locals(), globals()))
#     new_body = passer(body)
#     fun_tree = passer.parse(new_body)
#
#     _locals = dict(
#         zip(
#             ("_arg2", "_arg3", "_arg4", "_arg5"),
#             [cst.Integer(value=repr(i)) for i in (0, 0, 0, 0)],
#         )
#     )
#     sym_table = SymbolTable(_locals, {})
#     new_new_body = replace_symbols(fun_tree.body, sym_table)
#     open("new_new_body.py", "w").write(to_source(new_new_body))
#     fun_tree = fun_tree.with_deep_changes(fun_tree, body=new_new_body)
#     open("unrolled.py", "w").write(to_source(fun_tree))


def parse_range_call(rang):
    lo, hi, step = [r.value for r in rang.args]
    return hi


# class FindMaxRange(ast.NodeTransformer):
#     max_range = None
#
#     def visit_Call(self, node: Call) -> Any:
#         call_str = astor.code_gen.to_source(node)
#         if "ParFor" in call_str:
#             ranges = []
#             for range_call in node.keywords[0].value.elts:
#                 ranges.append(parse_range_call(range_call))
#             max_range = max(list(itertools.product(*ranges)))
#             if self.max_range is None:
#                 self.max_range = max_range
#             self.max_range = max(self.max_range, max_range)
#         self.generic_visit(node)
#         return node


class SetMaxRange(ast.NodeTransformer):
    def __init__(self, max_range):
        super(SetMaxRange, self).__init__()
        self.max_range = max_range

    def visit_Call(self, node: Call) -> Any:
        call_str = astor.code_gen.to_source(node)
        if "Forward" in call_str:
            node.keywords.append(
                keyword(arg="max_range", value=Str([i for i in self.max_range]))
            )
        self.generic_visit(node)
        return node


def transform_forward_py(fp, macs=False):
    new_tree = astor.parse_file(fp)
    # new_tree = RemoveMAC().visit(new_tree)
    new_tree = HoistGlobals().visit(new_tree)

    # new_tree = RemoveMulAdd().visit(new_tree)
    # if macs:
    #     new_tree = RemoveMulAddAndSubscript().visit(new_tree)

    # new_tree = SetMaxRange(max_range).visit(new_tree)
    # new_tree = RemoveMul().visit(new_tree)
    new_tree = RemoveIfExp().visit(new_tree)
    with open(f"{fp.replace('forward', 'forward_rewritten')}", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    parser.add_argument("--macs", action='store_true', default=False)
    args = parser.parse_args()
    transform_forward_py(args.fp, args.macs)
