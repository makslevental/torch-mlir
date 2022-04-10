import ast
import inspect
from ast import Assign, Mult, Add, BinOp, Name, Call

import astor
import numpy as np


# def double_to_hex(f):
#     return hex(struct.unpack("<I", struct.pack("<f", f))[0])
#
#
# def padAndFormatHex(h, numDigits):
#     if h is None:
#         return h
#     if h.endswith("L") or h.endswith("l"):
#         h = h[:-1]
#     # assumes it already starts with "0x"
#     while len(h) < numDigits + 2:
#         h = h[:2] + "0" + h[2:]
#     return h
#
#
# def returnFloatHex(f, h):
#     print("Content-type: text/xml\n")
#     print("<values>")
#     if f == "ERROR":
#         print("<float>ERROR</float>")
#     elif not isinstance(f, float):
#         print("<float>%s</float>" % f)
#     else:
#         print("<float>%g</float>" % f)
#     h = padAndFormatHex(h, 8)
#     print("<hex>%s</hex>" % h)
#     print("</values>")
#
#
# def returnDoubleHex(d, h):
#     print("<double>" + str(d) + "</double>")
#     h = padAndFormatHex(h, 16)
#     print("<hex>%s</hex>" % h)
#
#
# def isHexChar(c):
#     try:
#         i = int(c, 16)
#         return True
#     except:
#         return False
#
#
# def handleFloatToHex(f, swap=False):
#     h = FloatToHex.floattohex(f, swap)
#     h = str(hex(h)).lower()
#     h = padAndFormatHex(h, 8)
#     returnFloatHex(f, h)
#
#
# def handleHexToFloat(h, swap=False):
#     if not h.startswith("0x"):
#         h = "0x" + h
#     for c in h[2:]:
#         if not isHexChar(c):
#             returnFloatHex("ERROR", form.getfirst("hex"))
#             return
#     try:
#         i = int(h[2:], 16)
#         f = FloatToHex.hextofloat(i, swap)
#     except:
#         returnFloatHex("ERROR", form.getfirst("hex"))
#         return
#     returnFloatHex(f, h)
#
#
# def handleDoubleToHex(d, swap=False):
#     isNegative = False
#     dToPass = d
#     # weird handling for negative 0
#     if not swap and math.copysign(1, d) == -1:
#         isNegative = True
#         dToPass = -1.0 * d
#     h = FloatToHex.doubletohex(dToPass, swap)
#     h = str(hex(h)).lower()
#     h = padAndFormatHex(h, 16)
#     if isNegative:
#         h = h[0:2] + hex(int(h[2:3], 16) + 8)[2:] + h[3:]
#     return h
#
#
# def handleHexToDouble(h, swap=False):
#     if not h.startswith("0x"):
#         h = "0x" + h
#     for c in h[2:]:
#         if not isHexChar(c):
#             returnDoubleHex("ERROR", form.getfirst("hex"))
#             return
#     i = int(h[2:], 16)
#     d = FloatToHex.hextodouble(i, swap)
#     returnDoubleHex(d, h)


def format_cst(cst):
    return float(np.random.randint(1, 100000))
    return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


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
        print(f"{v} = fmul float {self}, {other}")
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        v = Val(f"(+ ({self}) ({other}))")
        print(f"{v} = fadd float {self}, {other}")
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
        print(f"{v} = fcmp ugt float {self}, {other}")
        return v

    def __str__(self):
        return f"%{self.var_id}"


class Constant(Val):
    def __str__(self):
        return f"{format_cst(float(self.name))}"


def Exp(val):
    v = Val(f"(exp ({val})")
    # print(f"{v} = call float @llvm.exp.f32(float {val})")
    print(f"{v} = call float @expf(float {val})")
    return v


class ArrayDecl:
    def __init__(self, var_name, *shape, input=False, output=False, globl=False):
        self.var_name = var_name
        self.curr_shape = shape
        self.prev_shape = shape
        self.registers = {}
        self.input = input
        self.output = output
        self.globl = globl

    def __getitem__(self, index):
        # print("load", self.var_name, index)
        index = self.idx_map(index)
        if index not in self.registers:
            assert self.input or self.globl, (self.var_name, index)
            v = Val(f"{self.var_name}_{'_'.join(map(str, index))}")
            v.var_id = v.name
            self.registers[index] = v
            if self.globl:
                shape = self.curr_shape
                typ = get_array_type(shape, ptr=False)
                idx = ", ".join([f"i64 {i}" for i in ([0] + list(index))])
                print(
                    f"%{v.name} = load float, float* getelementptr inbounds ({typ}, {typ}* @{self.var_name}, {idx}), align 4")

        return self.registers[index]

    def __setitem__(self, index, value):
        # print("store", self.var_name, index, value)
        # assert not self.input
        index = self.idx_map(index)
        self.registers[index] = value
        assert not self.input and not self.globl
        if self.output:
            # store i32 %1, i32* %out_r, align 4
            var_name = f"%{self.var_name}_{'_'.join(map(str, index))}"
            print(f"store float {value}, float* {var_name}, align 4")

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
        # print("load", cst)
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
    # print("declare float @llvm.exp.f32(float %a)")
    print("declare float @expf(float)")
    print("declare float @llvm.fmuladd.f32(float %a, float %b, float %c)")
    for glo in globals:
        print(glo)

    print(f"define void @forward({', '.join(args)}) {{\n")
    forward()
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
