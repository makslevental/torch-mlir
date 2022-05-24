from __future__ import annotations

import argparse
import ast
import inspect
import itertools
import sys
from ast import Assign, Mult, Add, BinOp, Name, Call, keyword, Str, IfExp, Compare
from typing import Tuple, Union, Dict, Any

import astor
import numpy as np


MAC_IDX = None

OUTPUT_ARRAYS = []

from dataclasses import dataclass, field


class Val:
    def __init__(self, name, val_id):
        self.name = name
        self.val_id = val_id

    def __repr__(self):
        return str(self.__class__.__name__)

    def __mul__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        m = MulInst(self, other)
        v = Val(f"(* {self} {other})", m)
        PES[CURRENT_PE].push_instructions(m)
        return v

    def __truediv__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        d = DivInst(self, other)
        v = Val(f"(/ {self} {other})", d)
        PES[CURRENT_PE].push_instructions(d)
        return v

    def __add__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        a = AddInst(self, other)
        v = Val(f"(+ {self} {other})", a)
        PES[CURRENT_PE].push_instructions(a)
        return v

    def __sub__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        s = SubInst(self, other)
        v = Val(f"(- {self} {other})", s)
        PES[CURRENT_PE].push_instructions(s)
        return v

    def __gt__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        g = GTInst(self, other)
        v = Val(f"(> {self} {other})", g)
        PES[CURRENT_PE].push_instructions(g)
        return v


ArrayIndex = Tuple[int]


class ArrayVal(Val):
    array: ArrayDecl
    def __init__(self, name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(name, val_id)
        self.array = array

    def __str__(self):
        return f"array{self.name}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"global{self.name}_{'_'.join(map(str, self.val_id))}"


PEIndex = Tuple[int]
@dataclass(frozen=True)
class _Instruction:
    pe_id: PEIndex = field(init=False, default_factory=lambda: CURRENT_PE)


@dataclass(frozen=True)
class NOP(_Instruction):
    pass


@dataclass(frozen=True)
class Bin(_Instruction):
    left: Val
    right: Val


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
    val: Val


@dataclass(frozen=True)
class ExpInst(_Instruction):
    val: Val


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


class Constant(Val):
    def __init__(self, cst_val):
        super().__init__("cst", cst_val)


def ReLU(*args):
    def op(arg):
        pe = PES[CURRENT_PE]
        r = ReLUInst(arg)
        v = Val(f"(relu {arg})", r)
        pe.push_instructions(r)
        return v

    return op


def Exp(*args):
    def op(arg):
        pe = PES[CURRENT_PE]
        r = ExpInst(arg)
        v = Val(f"(exp {arg})", r)
        pe.push_instructions(r)
        return v

    return op


def index_map(index, curr_shape, prev_shape):
    return tuple(
        np.unravel_index(
            np.ravel_multi_index(index, curr_shape), prev_shape
        )
    )


class ArrayDecl:
    def __init__(self, arr_name, *shape, input=False, output=False):
        self.arr_name = arr_name
        self.curr_shape = shape
        self.prev_shape = shape
        self.pe_index = shape
        self.registers = {}
        self.input = input
        self.output = output

    def __getitem__(self, index: ArrayIndex):
        global PES
        try:
            index = self.idx_map(index)
        except ValueError:
            index = (-1, -1, -1, -1)

        if index not in self.registers:
            if not self.input:
                v = Constant("0.0")
            else:
                v = ArrayVal(f"{self.arr_name}", index, self)
            self.registers[index] = v

        v = self.registers[index]
        return v

    def __setitem__(self, index, value):
        global PES
        try:
            index = self.idx_map(index)
        except ValueError:
            index = (-1, -1, -1, -1)
        assert not self.input
        self.registers[index] = value

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class GlobalArray:
    def __init__(self, name, global_name, global_array):
        self.name = name
        self.global_name = global_name
        self.global_array = global_array
        self.curr_shape = global_array.shape
        self.csts = {}

    def __getitem__(self, index: ArrayIndex):
        v = GlobalArrayVal(self.name, index, self)
        return v


def ParFor(body, ranges):
    global PES, CURRENT_PE
    pes_run = set()
    num_insts = None
    for i, idx in enumerate(sorted(itertools.product(*ranges))):
        CURRENT_PE = i
        cur_num_insts = PES[CURRENT_PE].num_instructions()
        body(*idx)
        pes_run.add(i)
        if num_insts is None:
            num_insts = PES[CURRENT_PE].num_instructions() - cur_num_insts

    for pe_idx, pe in PES.items():
        if pe_idx not in pes_run:
            for _ in range(num_insts):
                pe.push_nop()


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
        typ += "float"
        for _ in shape:
            typ += "]"
        if ptr:
            typ += "*"
    else:
        typ = f"{np.prod(shape)} x float"
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
            relu_inst = Call(func=Name(id="ReLU"), args=self.body_args, keywords=[])
            return Assign(
                targets=node.targets,
                value=Call(func=relu_inst, args=self.subs[node], keywords=[]),
                type_comment=None,
            )
        else:
            return node

    def visit_AugAssign(self, node):
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


def transform_forward_py(fp, max_range):
    code_ast = astor.parse_file(fp)
    # new_tree = RemoveMAC().visit(code_ast)
    # new_tree = RemoveMulOrAdd().visit(code_ast)
    new_tree = SetMaxRange(max_range).visit(code_ast)
    # new_tree = RemoveMul().visit(code_ast)
    new_tree = RemoveIfExp().visit(new_tree)
    with open(f"{fp.replace('forward', 'forward_rewritten')}", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


if __name__ == "__main__":
    fp = sys.argv[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('fp')
    parser.add_argument('--max_range', nargs='+', type=int)
    args = parser.parse_args()
    print(args)
    max_range = tuple(args.max_range)
    transform_forward_py(args.fp, max_range)
