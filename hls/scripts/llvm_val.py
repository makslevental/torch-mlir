import numpy as np


def format_cst(cst):
    return float(np.random.randint(1, 100000))
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


VAR_COUNT = 0


class LLVMVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.var_id = f"val_{VAR_COUNT}"
        VAR_COUNT += 1

    def __mul__(self, other):
        # <result> = fmul float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(* ({self}) ({other}))")
        print(f"{v} = fmul float {self}, {other}")
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float {other}, float 0.0)"
        # )
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(+ ({self}) ({other}))")
        print(f"{v} = fadd float {self}, {other}")
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float 1.0, float {other})"
        # )
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(- ({self}) ({other}))")
        print(f"{v} = fsub float {self}, {other}")
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(/ ({self}) ({other}))")
        print(f"{v} = fdiv float {self}, {other}")
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(> ({self}) ({other}))")
        # print(f"{v} = fcmp ugt float {self}, {other}")
        return v

    def __str__(self):
        return f"%{self.var_id}"


class LLVMConstant(LLVMVal):
    def __init__(self, name):
        super(LLVMConstant, self).__init__(name)
        self._fmt = f"{format_cst(float(self.name))}"

    def __str__(self):
        return self._fmt


def Exp(val):
    v = LLVMVal(f"(exp ({val})")
    print(f"{v} = call float @expf(float {val})")
    return v


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = LLVMConstant(v)
    v = LLVMVal(f"(fmuladd ({a}) ({b}) ({c})")
    print(
        f"{v} = call float @llvm.fmuladd.f32(float {inps[0]}, float {inps[1]}, float {inps[2]})"
    )
    return v


def make_args_globals(Args):
    from mlir_ops import get_array_type
    from mlir_ops import ArrayDecl

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
                typ = get_array_type(arg.curr_shape, ptr=False)
                globals.append(
                    f"@{arg.var_name} = common global {typ} zeroinitializer, align 4"
                )

    return args, globals


def LLVMForward(Args, OUTPUT_ARRAYS, forward):
    args, globals = make_args_globals(Args)

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
