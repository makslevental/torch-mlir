from __future__ import annotations

import _ast
import argparse
import ast
import glob
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
                        func=Name(id="FMAC"),
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
            assigns = [
                (i, b)
                for i, b in enumerate(node.body)
                if isinstance(b, Assign)
                and isinstance(b.value, Call)
                and isinstance(b.value.func, Name)
                and b.value.func.id == "GlobalArray"
            ]
            for i, a in reversed(assigns):
                del node.body[i]
                node.args.args.append(a.targets[0].id)
                node.args.defaults.append(a.value)
        return node


class RemoveValSemCopies(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "forward":
            arg_names = [
                arg.arg if isinstance(arg, ast.arg) else arg for arg in node.args.args
            ]
            all_copies = [
                (i, b)
                for i, b in enumerate(node.body)
                if isinstance(b, _ast.FunctionDef) and len(b.body) == 2
            ]
            for i, copy in reversed(all_copies):
                src = copy.body[0].value.value
                if src.id in arg_names:
                    continue
                dst = copy.body[1].targets[0].value
                assert (
                    isinstance(node.body[i + 1].value, Call)
                    and node.body[i + 1].value.func.id == "ParFor"
                )
                node.body[i] = ast.Expr(
                    value=Call(
                        func=Name(id="Copy"),
                        args=[dst, src],
                        keywords=[
                            keyword(
                                arg="valsem", value=_ast.Constant(kind=None, value=True)
                            )
                        ],
                    )
                )
                del node.body[i + 1]
        return node


def parse_range_call(rang):
    lo, hi, step = [r.value for r in rang.args]
    return hi


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


class HandleUnrolledLoops(ast.NodeTransformer):
    has_fma = False

    def visit_FunctionDef(self, node):
        args = [arg if isinstance(arg, ast.arg) else arg for arg in node.args.args if arg.arg != "file"]
        file_arg = [arg if isinstance(arg, ast.arg) else arg for arg in node.args.args if arg.arg == "file"]
        assert file_arg
        file_arg = file_arg[0]

        arg_names = [arg.arg for arg in args]
        all_binops = [
            (i, b)
            for i, b in enumerate(node.body)
            if isinstance(b, _ast.Assign) and isinstance(b.value, _ast.BinOp)
        ]
        for i, (_, assign) in enumerate(all_binops):
            if isinstance(assign.value.op, _ast.Mult) and i + 1 < len(all_binops):
                _, next_assign = all_binops[i + 1]
                if isinstance(next_assign.value.op, _ast.Add):
                    self.has_fma = True
                    break

        if self.has_fma:
            stuff_to_delete = []
            all_subscript_assigns = [
                (i, b)
                for i, b in enumerate(node.body)
                if isinstance(b, _ast.Assign)
                and isinstance(b.targets[0], _ast.Subscript)
            ]
            assign_target = all_subscript_assigns[0][1].targets[0].value
            all_subscript_loads = [
                (i, b)
                for i, b in enumerate(node.body)
                if isinstance(b, _ast.Assign)
                and isinstance(b.value, _ast.Subscript)
                and b.value.value.id == assign_target.id
            ]
            for i, (j, assign) in enumerate(all_subscript_assigns[0:-1]):
                stuff_to_delete.append(j)
                k, load = all_subscript_loads[i + 1]
                name_to_replace = load.targets[0].id
                assert node.body[k + 2].value.left.id == name_to_replace
                node.body[k + 2].value.left = assign.value
                stuff_to_delete.append(all_subscript_loads[i + 1][0])

            for j in reversed(sorted(stuff_to_delete)):
                del node.body[j]

            fma = Name(id="fma")
            node.body.insert(
                0,
                Assign(
                    targets=[fma],
                    value=Call(func=Name(id="FMAC"), args=args, keywords=[
                        keyword(arg="file", value=file_arg)
                    ]),
                    type_comment=None,
                ),
            )
            for i, nod in enumerate(node.body):
                if (
                    isinstance(nod, _ast.Assign)
                    and isinstance(nod.value, _ast.BinOp)
                    and nod.value.left.id not in arg_names
                ):
                    if not isinstance(nod.targets, list):
                        targets = [nod.targets]
                    else:
                        targets = nod.targets

                    if isinstance(nod.value.op, _ast.Mult):
                        node.body[i] = Assign(
                            targets=targets,
                            value=Call(
                                func=Name(id="fma.Mul"),
                                args=[nod.value.left, nod.value.right],
                                keywords=[],
                            ),
                            type_comment=None,
                        )
                    elif isinstance(nod.value.op, _ast.Add):
                        node.body[i] = Assign(
                            targets=targets,
                            value=Call(
                                func=Name(id="fma.Add"),
                                args=[nod.value.left, nod.value.right],
                                keywords=[],
                            ),
                            type_comment=None,
                        )
            assert node.body[-1].targets[0].value.id == assign_target.id
            node.body[-1] = Assign(
                targets=[node.body[-1].targets[0]],
                value=Call(func=Name(id="fma.Result"), args=[], keywords=[]),
                type_comment=None,
            )

        return node


class ReplaceHandleUnrolledLoops(ast.NodeTransformer):
    def __init__(self, handled_loops):
        self.handled_loops = handled_loops

    def visit_FunctionDef(self, node):
        if node.decorator_list:
            apply_n = node.decorator_list[0].keywords[0].value.value
            if apply_n in self.handled_loops:
                node.body = self.handled_loops[apply_n].body[0].body
            node.decorator_list = []

        self.generic_visit(node)
        return node


def handle_unrolled_loops(fp):
    handled_loops = {}
    for unrolled_loop_fp in glob.glob(fp.replace("forward.py", ".ast_tools") + "/*.py"):
        new_tree = astor.parse_file(unrolled_loop_fp)
        han = HandleUnrolledLoops()
        new_tree = han.visit(new_tree)
        if han.has_fma:
            handled_loops[unrolled_loop_fp.rsplit("/", 1)[1]] = new_tree
            with open(
                f"{unrolled_loop_fp.replace('apply', 'apply_rewritten')}", "w"
            ) as f:
                f.write(astor.code_gen.to_source(new_tree))

    new_tree = astor.parse_file(fp)
    new_tree = ReplaceHandleUnrolledLoops(handled_loops).visit(new_tree)
    new_fp = f"{fp.replace('.py', '_unrolled.py')}"
    with open(new_fp, "w") as f:
        f.write(astor.code_gen.to_source(new_tree))
    return new_fp


def transform_forward_py(fp):
    new_tree = astor.parse_file(fp)
    new_tree = HoistGlobals().visit(new_tree)
    # new_tree = RemoveValSemCopies().visit(new_tree)
    new_tree = RemoveIfExp().visit(new_tree)
    new_fp = f"{fp.replace('.py', '_rewritten.py')}"
    with open(new_fp, "w") as f:
        f.write(astor.code_gen.to_source(new_tree))
    return new_fp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    args = parser.parse_args()
    print(args.fp)
    new_fp = handle_unrolled_loops(args.fp)
    transform_forward_py(new_fp)
