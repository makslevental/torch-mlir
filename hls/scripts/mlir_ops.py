import ast
import inspect
from ast import Assign, Mult, Add, BinOp, Name, Call

import astor
import numpy as np


def format_cst(cst):
    return float(np.random.randint(1, 100000))
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


var_count = 0


class Val:
    def __init__(self, name):
        global var_count
        self.name = f"{name}"
        self.var_id = f"val_{var_count}"
        var_count += 1

    def __mul__(self, other):
        # <result> = fmul float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(* ({self}) ({other}))")
        # print(f"{v} = fmul float {self}, {other}")
        print(f"{v} = call float @llvm.fmuladd.f32(float {self}, float {other}, float 0.0)")
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(+ ({self}) ({other}))")
        # print(f"{v} = fadd float {self}, {other}")
        print(f"{v} = call float @llvm.fmuladd.f32(float {self}, float 1.0, float {other})")
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(- ({self}) ({other}))")
        print(f"{v} = fsub float {self}, {other}")
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(/ ({self}) ({other}))")
        print(f"{v} = fdiv float {self}, {other}")
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(> ({self}) ({other}))")
        # print(f"{v} = fcmp ugt float {self}, {other}")
        return v

    def __str__(self):
        return f"%{self.var_id}"


class Constant(Val):
    def __init__(self, name):
        super(Constant, self).__init__(name)
        self._fmt = f"{format_cst(float(self.name))}"

    def __str__(self):
        return self._fmt


def Exp(val):
    v = Val(f"(exp ({val})")
    print(f"{v} = call float @expf(float {val})")
    return v


OUTPUT_ARRAYS = []


class ArrayDecl:
    def __init__(self, var_name, *shape, input=False, output=False, globl=False):
        global OUTPUT_ARRAYS
        self.var_name = var_name
        self.curr_shape = shape
        self.prev_shape = shape
        self.registers = {}
        self.input = input
        self.output = output
        if output:
            OUTPUT_ARRAYS.append(self)
        self.globl = globl

    def __getitem__(self, index):
        index = self.idx_map(index)
        if index not in self.registers:
            assert self.input or self.globl, (self.var_name, index)
            v = Val(f"{self.var_name}_{'_'.join(map(str, index))}")
            v.var_id = v.name
            self.registers[index] = v
            # if self.globl:
            #     shape = self.curr_shape
            #     typ = get_array_type(shape, ptr=False)
            #     idx = ", ".join([f"i64 {i}" for i in ([0] + list(index))])
            #     print(
            #         f"%{v.name} = load float, float* getelementptr inbounds ({typ}, {typ}* @{self.var_name}, {idx}), align 4")

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
    def __init__(self, var_name, global_name, global_array):
        self.var_name = var_name
        self.global_name = global_name
        self.global_array = global_array
        self.csts = {}

    def __getitem__(self, index):
        if index not in self.csts:
            self.csts[index] = Constant(self.global_array[index])
        cst = self.csts[index]
        return cst

    def __setitem__(self, key, value):
        raise Exception("wtfbbq")

    def reshape(self, shape):
        raise Exception("wtfbbq")


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = Constant(v)
    v = Val(f"(fmuladd ({a}) ({b}) ({c})")
    print(
        f"{v} = call float @llvm.fmuladd.f32(float {inps[0]}, float {inps[1]}, float {inps[2]})"
    )
    return v


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
        typ += f"]"
    if ptr:
        typ += "*"
    return typ


def Forward(forward):
    Args = get_default_args(forward)
    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            if arg.input:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(f"float %{arg.var_name}_{'_'.join(map(str, index))}")
            elif arg.output:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(f"float* %{arg.var_name}_{'_'.join(map(str, index))}")
            elif arg.globl:
                # @src32 = common global [16 x float], align 4
                typ = get_array_type(arg.curr_shape, ptr=False)
                globals.append(
                    f"@{arg.var_name} = common global {typ} zeroinitializer, align 4"
                )

    print('source_filename = "LLVMDialectModule"')
    print("declare float @expf(float)")
    print("declare float @llvm.fmuladd.f32(float %a, float %b, float %c)")
    for glo in globals:
        print(glo)

    print(f"define void @forward({', '.join(args)}) {{\n")
    forward()
    for arr in OUTPUT_ARRAYS:
        for index, value in arr.registers.items():
            var_name = f"%{arr.var_name}_{'_'.join(map(str, index))}"
            print(f"store float {value}, float* {var_name}, align 4")

    print("ret void")
    print("}")


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


def transform_forward_py():
    code_ast = astor.parse_file("forward.py")
    new_tree = RemoveMulAdd().visit(code_ast)
    with open("forward_rewritten.py", "w") as f:
        f.write(astor.code_gen.to_source(new_tree))


def test_hex():
    print(format_cst(1000 + np.random.randn(1)[0]))


if __name__ == "__main__":
    transform_forward_py()
