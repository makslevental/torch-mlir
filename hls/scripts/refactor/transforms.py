import argparse
import ast
from ast import Assign, Mult, Add, BinOp, Name, Call, IfExp, Compare

import astor

from torch_mlir._mlir_libs._mlir.ir import Context, Module, OpView, FunctionType
from hls.mlir.python_extension.hls._mlir_libs.hls_utils import get_val_identifier


class RemoveMAC(ast.NodeTransformer):
    body_args = []
    has_fma = False
    final_assign = None

    def visit_FunctionDef(self, node):
        if node.name == "body":
            self.body_args = node.args.args
            self.generic_visit(node)
            if self.has_fma and node.name == "body":
                fma = Name(id="fma")
                node.body.insert(
                    0,
                    Assign(
                        targets=[fma],
                        value=Call(
                            func=Name(id="FMAC"),
                            args=self.body_args[:-1],
                            keywords=[],
                        ),
                        type_comment=None,
                    ),
                )
                node.body.append(
                    Assign(
                        targets=[self.final_assign],
                        value=Call(func=Name(id="fma.Result"), args=[], keywords=[]),
                        type_comment=None,
                    )
                )
                self.has_fma = False

        self.generic_visit(node)
        return node

    def visit_For(self, node):
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
                    self.has_fma = True
                    self.final_assign = assigns[-1].targets[0]
                    node.body[i] = Assign(
                        targets=assign1.targets,
                        value=Call(
                            func=Name(id="fma.Mul"),
                            args=[assign1.value.left, assign1.value.right],
                            keywords=[],
                        ),
                        type_comment=None,
                    )
                    node.body[i + 1] = Assign(
                        targets=assign2.targets,
                        value=Call(
                            func=Name(id="fma.Add"),
                            args=[assign2.value.left, assign2.value.right],
                            keywords=[],
                        ),
                        type_comment=None,
                    )

        self.generic_visit(node)
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


class HoistGlobals(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == "forward":
            assigns = [
                (i, b)
                for i, b in enumerate(node.body)
                if isinstance(b, Assign)
                and isinstance(b.value, Call)
                and isinstance(b.value.func, Name)
                and b.value.func.id == "GlobalMemRef"
            ]
            for i, a in reversed(assigns):
                del node.body[i]
                node.args.args.append(a.targets[0].id)
                node.args.defaults.append(a.value)
        return node


def traverse_op_region_block_iterators(op, handler):
    for i, region in enumerate(op.regions):
        for j, block in enumerate(region):
            for k, child_op in enumerate(block):
                handler(child_op)
                traverse_op_region_block_iterators(child_op, handler)


def parse_attrs_to_dict(attrs):
    d = {}
    for named_attr in attrs:
        if named_attr.name in {"lpStartTime", "value"}:
            d[named_attr.name] = ast.literal_eval(
                str(named_attr.attr).split(":")[0].strip()
            )
        else:
            d[named_attr.name] = ast.literal_eval(str(named_attr.attr))
    return d


def parse_mlir_module(module_str):
    ctx = Context()
    ctx.allow_unregistered_dialects = True
    module = Module.parse(
        module_str,
        ctx,
    )
    op = module.operation

    def handler(mlir_op):
        if not isinstance(mlir_op, OpView) or (
            hasattr(mlir_op, "type") and isinstance(mlir_op.type, FunctionType)
        ):
            return

        if list(mlir_op.results):
            ident = get_val_identifier(mlir_op.result._CAPIPtr)
            print(ident, mlir_op.operation.name, parse_attrs_to_dict(mlir_op.attributes))

    traverse_op_region_block_iterators(op, handler)


def transform_forward_py(fp):
    new_tree = astor.parse_file(fp)
    new_tree = HoistGlobals().visit(new_tree)
    new_tree = RemoveIfExp().visit(new_tree)
    new_tree = RemoveMAC().visit(new_tree)

    new_fp = f"{fp.replace('.py', '_rewritten.py')}"
    with open(new_fp, "w") as f:
        f.write(astor.code_gen.to_source(new_tree))
    return new_fp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    parser.add_argument("--py", action="store_true")
    parser.add_argument("--mlir", action="store_true")
    args = parser.parse_args()
    if args.py:
        transform_forward_py(args.fp)
    if args.mlir:
        sched_module_str = open(args.fp).read()
        parse_mlir_module(sched_module_str)
