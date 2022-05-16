import ast
import inspect
import warnings
from ast import Assign, Mult, Add, BinOp, Name, Call, keyword, Str, IfExp, Compare

import astor
import numpy as np

# from ast_tools.passes import apply_passes
from llvm_val import LLVMForward, LLVMVal, LLVMConstant
# from verilog_val import VerilogWire, VerilogConstant, VerilogForward

MAC_IDX = None

OUTPUT_ARRAYS = []


class ArrayDecl:
    def __init__(
            self,
            var_name,
            *shape,
            input=False,
            output=False,
            globl=False,
            val_cons=LLVMVal,
    ):
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
            if not self.input:
                return LLVMConstant("0.0")
            v = self.val_cons(f"{self.var_name}_{'_'.join(map(str, index))}")
            v.var_id = v.name
            self.registers[index] = v

        return self.registers[index]

    def __setitem__(self, index, value):
        index = self.idx_map(index)
        assert not self.input and not self.globl
        if self.val_cons == LLVMVal:
            if index in self.registers:
                var_name = f"%{self.var_name}_{'_'.join(map(str, index))}"
                # print(f"store float {value}, float* {var_name}, align 4")
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
    def __init__(self, var_name, global_name, global_array, cst_cons=LLVMConstant):
        # @cx = global { float, float } { float 1.000000e+00, float 9.900000e+01 }, align 4
        # @_ZL5arg_1 = internal constant [10 x [10 x [10 x [10 x float]]]] zeroinitializer, align 16
        self.var_name = var_name
        self.global_name = global_name
        self.global_array = global_array
        self.curr_shape = global_array.shape
        self.cst_cons = cst_cons
        self.csts = {}

    def __getitem__(self, index):
        # if index not in self.csts:
        #     self.csts[index] = self.cst_cons(self.global_array[index])
        # cst = self.csts[index]
        v = LLVMVal(self.var_name)
        idx = [str(np.ravel_multi_index(index, self.curr_shape))]
        v.var_id = f"glob_{self.var_name}_idx_{'_'.join(idx)}"
        return v
        # return cst

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


def Forward(forward):
    Args = get_default_args(forward)
    LLVMForward(Args, OUTPUT_ARRAYS, forward)


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
                    self.subs[assign] = (
                        assign.value.left,
                        assign.value.right,
                    )

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
                value=Call(
                    func=relu_inst,
                    args=self.subs[node],
                    keywords=[],
                ),
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
    code_ast = astor.parse_file("forward.py")
    new_tree = RemoveMAC().visit(code_ast)
    new_tree = RemoveMulOrAdd().visit(new_tree)
    new_tree = RemoveIfExp().visit(new_tree)
    with open("forward_rewritten.py", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


if __name__ == "__main__":
    transform_forward_py()
