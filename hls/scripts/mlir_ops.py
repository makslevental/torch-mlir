import ast
import inspect
from ast import Assign, Mult, Add, BinOp, Name, Call

import astor
import numpy as np

from verilog_val import VerilogForward, VerilogWire, VerilogConstant

MAC_IDX = None

OUTPUT_ARRAYS = []


class ArrayDecl:
    def __init__(self, var_name, *shape, input=False, output=False, globl=False, val_cons=VerilogWire):
        global OUTPUT_ARRAYS
        self.var_name = var_name
        self.curr_shape = shape
        self.prev_shape = shape
        self.registers = {}
        self.input = input
        self.output = output
        self.val_cons = val_cons
        if output:
            OUTPUT_ARRAYS.append(self)
        self.globl = globl

    def __getitem__(self, index):
        index = self.idx_map(index)
        if index not in self.registers:
            assert self.input or self.globl, (self.var_name, index)
            v = self.val_cons(f"{self.var_name}_{'_'.join(map(str, index))}")
            v.var_id = v.name
            self.registers[index] = v

        return self.registers[index]

    def __setitem__(self, index, value):
        index = self.idx_map(index)
        assert not self.input and not self.globl
        # if index in self.registers:
        #     var_name = f"%{self.var_name}_{'_'.join(map(str, index))}"
        #     print(f"store float {value}, float* {var_name}, align 4")
        self.registers[index] = value

    def idx_map(self, index):
        return np.unravel_index(
            np.ravel_multi_index(index, self.curr_shape), self.prev_shape
        )

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class Global:
    def __init__(self, var_name, global_name, global_array, cst_cons=VerilogConstant):
        self.var_name = var_name
        self.global_name = global_name
        self.global_array = global_array
        self.cst_cons = cst_cons
        self.csts = {}

    def __getitem__(self, index):
        if index not in self.csts:
            self.csts[index] = self.cst_cons(self.global_array[index])
        cst = self.csts[index]
        return cst

    def __setitem__(self, key, value):
        raise Exception("wtfbbq")

    def reshape(self, shape):
        raise Exception("wtfbbq")


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_array_type(shape, ptr=True):
    typ = ""
    for s in shape:
        typ += f"[{s} x "
    typ += "float"
    for _ in shape:
        typ += "]"
    if ptr:
        typ += "*"
    return typ


def Forward(forward):
    Args = get_default_args(forward)
    VerilogForward(Args, OUTPUT_ARRAYS, forward)


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
                value=Call(func=Name(id="MAC"), args=self.subs[node], keywords=[]),
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


def transform_forward_py():
    code_ast = astor.parse_file("braggnn.py")
    new_tree = RemoveMAC().visit(code_ast)
    with open("braggnn_forward_rewritten.py", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


if __name__ == "__main__":
    transform_forward_py()
