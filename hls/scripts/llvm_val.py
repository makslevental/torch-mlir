import io
import itertools
import sys
from collections import defaultdict, deque

import numpy as np


def format_cst(cst):
    return cst
    # return float(np.random.randint(1, 100000))
    # return cst
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


VAR_COUNT = 0

FILE = open("forward.ll", "w")

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
        if "-1" in f"{self}":
            print(f"{v} = fmul float 1.0, {other}", file=FILE)
        elif "-1" in f"{other}":
            print(f"{v} = fmul float {self}, 1.0", file=FILE)
        else:
            print(f"{v} = fmul float {self}, {other}", file=FILE)
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float {other}, float 0.0)"
        # )
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(+ ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"{v} = fadd float 0.0, {other}", file=FILE)
        elif "-1" in f"{other}":
            print(f"{v} = fadd float {self}, 0.0", file=FILE)
        else:
            print(f"{v} = fadd float {self}, {other}", file=FILE)
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float 1.0, float {other})"
        # )
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(- ({self}) ({other}))")
        print(f"{v} = fsub float {self}, {other}", file=FILE)
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(/ ({self}) ({other}))")
        print(f"{v} = fdiv float {self}, {other}", file=FILE)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(> ({self}) ({other}))")
        print(f"{v} = fcmp ugt float {self}, {other}", file=FILE)
        return v

    def __str__(self):
        return f"%{self.var_id}"


class LLVMConstant(LLVMVal):
    def __init__(self, name):
        super(LLVMConstant, self).__init__(name)
        self._fmt = f"{format_cst(self.name)}"

    def __str__(self):
        return self._fmt


def Exp(val):
    v = LLVMVal(f"(exp ({val})")
    print(f"{v} = call float @expf(float {val})", file=FILE)
    return v


MAC_QUEUES = defaultdict(deque)


class MMAC:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = LLVMConstant(v)
            if "glob" in v.var_id:
                v.var_id = f"mac_{self.id}_{v.var_id}"
                self.csts.append(v.var_id)

        if kwargs["type"] == "MulAdd":
            a, b, c = args
            v = LLVMVal(f"(mac ({a} {b} {c})")
            print(f"{v} = call float @llvm.fmuladd.f32(float {a}, float {b}, float {c})", file=FILE)
            return v
        else:
            a, b = args
            if kwargs["type"] == "Add":
                return a + b
            elif kwargs["type"] == "Mult":
                return a * b

MACS = {}


def MAC(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0 : len(idx)] = idx
        idx = tuple(_idx)

    if idx not in MACS:
        MACS[idx] = MMAC(idx)
    mac = MACS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


class RReLU:
    def __init__(self, idx):
        self.idx = idx
        self.id = "_".join(map(str, idx))
        self.val = LLVMVal(self.id)
        self.val.var_id = self.id

    def __call__(self, val):
        v = LLVMVal(f"(relu ({val})")
        print(f"{v} = call float @expf(float {val})", file=FILE)
        return v


RELUS = {}


def ReLU(*idx):
    if idx not in RELUS:
        RELUS[idx] = RReLU(idx)
    relu = RELUS[idx]

    def op(val):
        return relu(val)

    return op


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


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
    from mlir_ops import ArrayDecl, Global

    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
            args.append(f"{typ}* noalias %{arg.var_name}")
            # if arg.input:
            #     for index in np.ndindex(*arg.curr_shape):
            # elif arg.output:
            #     for index in np.ndindex(*arg.curr_shape):
            #         args.append(f"float* %{arg.var_name}_{'_'.join(map(str, index))}")
            # elif arg.globl:
            #     typ = get_array_type(arg.curr_shape, ptr=False)
            #     globals.append(
            #         f"@{arg.var_name} = common global {typ} zeroinitializer, align 4"
            #     )
        elif isinstance(arg, Global):
            globals.append(arg)

    return args, globals


def LLVMForward(Args, OUTPUT_ARRAYS, forward):
    global FILE
    from mlir_ops import get_array_type
    from mlir_ops import ArrayDecl

    args, globals = make_args_globals(Args)

    print('source_filename = "LLVMDialectModule"', file=FILE)
    print("declare float @expf(float)", file=FILE)
    print("declare void @_ssdm_op_SpecResource(...)", file=FILE)
    # print("declare float @relu(float)", file=FILE)
    print("declare float @llvm.fmuladd.f32(float %a, float %b, float %c)", file=FILE)
    print(f"define void @forward({', '.join(args)}) {{\n", file=FILE)
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
            # shape = arg.curr_shape if len(arg.curr_shape) > 1 else (arg.curr_shape,)
            for idx in itertools.product(*[range(a) for a in arg.curr_shape]):
                idx = list(map(str, idx))
                print(
                    f"%{arg.var_name}_{'_'.join(idx)}_gep = getelementptr {typ}, {typ}* %{arg.var_name}, i32 0, {', '.join([f'i32 {i}' for i in idx])}", file=FILE
                    # f"%mac_{mac.id}_{i}_gep = getelementptr {typ}, {typ}* %mac_{mac.id}, i16 0, i16 {i}", file=OLD_FILE
                )
                if arg.input:
                    print(
                        # f"%glob_{glo.var_name}_idx_{'_'.join(idx)} = load float, float* getelementptr inbounds ({typ}, {typ}* %glob_{glo.var_name}, i64 0, {', '.join([f'i64 {i}' for i in idx])})"
                        f"%{arg.var_name}_{'_'.join(idx)} = load float, float* %{arg.var_name}_{'_'.join(idx)}_gep", file=FILE
                    )

    OLD_FILE = FILE
    FILE = io.StringIO()
    forward()

    for mac_idx, mac in MACS.items():
        typ = get_array_type((len(mac.csts),))
        print(f"%mac_{mac.id} = alloca [{typ}], align 512", file=OLD_FILE)
        print(f"call void (...) @_ssdm_op_SpecResource([{typ}]* %mac_{mac.id}, i64 666, i64 22, i64 2)", file=OLD_FILE)

    for mac_idx, mac in MACS.items():
        typ = get_array_type((len(mac.csts),), nd=False, ptr=False)
        for i, cst in enumerate(mac.csts):
            print(
                # f"%glob_{glo.var_name}_idx_{'_'.join(idx)} = load float, float* getelementptr inbounds ({typ}, {typ}* %glob_{glo.var_name}, i64 0, {', '.join([f'i64 {i}' for i in idx])})"
                f"%mac_{mac.id}_{i}_gep = getelementptr [{typ}], [{typ}]* %mac_{mac.id}, i16 0, i16 {i}", file=OLD_FILE
            )
            print(
                # f"%glob_{glo.var_name}_idx_{'_'.join(idx)} = load float, float* getelementptr inbounds ({typ}, {typ}* %glob_{glo.var_name}, i64 0, {', '.join([f'i64 {i}' for i in idx])})"
                f"%{cst} = load float, float* %mac_{mac.id}_{i}_gep", file=OLD_FILE
            )

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for arr in OUTPUT_ARRAYS:
        for index, value in arr.registers.items():
            var_name = f"%{arr.var_name}_{'_'.join(map(str, index))}"
            print(f"store float {value}, float* {var_name}_gep, align 4", file=OLD_FILE)

    print("ret void", file=OLD_FILE)
    print("}", file=OLD_FILE)



# declare void @_ssdm_op_SpecResource(...)
#
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v85, i64 666, i64 23, i64 2)
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v86, i64 666, i64 20, i64 -1)
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v87, i64 666, i64 22, i64 2)
#
# volatile int8_t v85[1][3][34][34];	// L71
# #pragma HLS BIND_STORAGE variable=v85 type=RAM_2P impl=LUTRAM latency=2
# int8_t v86[1][3][34][34];	// L71
# #pragma HLS BIND_STORAGE variable=v86 type=RAM_1P impl=URAM
# int8_t v87[1][3][34][34];	// L7
# #pragma HLS BIND_STORAGE variable=v87 type=RAM_2P impl=BRAM latency=2V