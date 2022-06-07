from __future__ import annotations

import _ast
import argparse
import ast
from _ast import (
    Assign,
    BinOp,
    Mult,
    Add,
    Call,
    Name,
    keyword,
    Compare,
    IfExp,
    Subscript,
)
from ast import Str
from typing import Any

import astor


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


class RemoveMulAddAndSubscript(ast.NodeTransformer):
    subs = {}
    dels = set()

    def visit_For(self, node):
        self.generic_visit(node)
        assigns = [b for b in node.body if isinstance(b, Assign)]
        for i in range(len(assigns) - 2):
            assign1, assign2, assign3 = assigns[i: i + 3]
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


def transform_forward_py(fp, max_range=None, macs=False):
    new_tree = astor.parse_file(fp)
    # new_tree = RemoveMAC().visit(new_tree)
    new_tree = RemoveMulAdd().visit(new_tree)
    if macs:
        new_tree = RemoveMulAddAndSubscript().visit(new_tree)
    # new_tree = SetMaxRange(max_range).visit(new_tree)
    # new_tree = RemoveMul().visit(new_tree)
    new_tree = RemoveIfExp().visit(new_tree)
    with open(f"{fp.replace('forward', 'forward_rewritten')}", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    parser.add_argument("--max_range", nargs="+", type=int)
    parser.add_argument("--macs", action="store_true", default=False)
    args = parser.parse_args()
    if args.max_range:
        max_range = tuple(args.max_range)
    else:
        max_range = None
    transform_forward_py(args.fp, max_range, args.macs)
