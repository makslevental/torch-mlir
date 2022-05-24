from __future__ import annotations

import io
import itertools
import re
import struct
import sys
from collections import defaultdict, deque
from typing import Tuple

import networkx as nx
import numpy as np

from hls.scripts.mlir_ops import get_default_args, get_array_type, index_map


def float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def format_cst(cst):
    # return float_to_hex(float(cst))
    return float(np.random.randint(1, 100000))
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


ArrayIndex = Tuple[int]


class ArrayVal(LLVMVal):
    array: ArrayDecl

    def __init__(self, name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(name)
        self.array = array
        self.var_id = "_".join(map(str, val_id))

    def __str__(self):
        return f"%{self.name}_{self.var_id}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"%{self.name}_{self.var_id}"


class GlobalArray:
    def __init__(self, name, global_name, global_array):
        self.name = name
        self.global_name = global_name
        self.global_array = global_array
        self.curr_shape = global_array.shape
        self.csts = {}

    def __getitem__(self, index: ArrayIndex):
        if index not in self.csts:
            v = GlobalArrayVal(self.global_name, index, self)
            self.csts[index] = v
            print(f"{v} = fptrunc double {format_cst('')} to float", file=FILE)
        return self.csts[index]


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
                v = LLVMConstant("0.0")
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
            print(
                f"{v} = call float @llvm.fmuladd.f32(float {a}, float {b}, float {c})",
                file=FILE,
            )
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
        print(f"{v} = call float @relu(float {val})", file=FILE)
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
    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
            args.append(f"{typ}* noalias %{arg}")

    return args, globals


def LLVMForward(input, output, forward, processing_elts, max_range):
    global FILE
    print('source_filename = "LLVMDialectModule"', file=FILE)
    print("declare float @expf(float)", file=FILE)
    print("declare void @_ssdm_op_SpecResource(...)", file=FILE)
    print("declare float @relu(float)", file=FILE)
    print("declare float @llvm.fmuladd.f32(float %a, float %b, float %c)", file=FILE)
    inputs = [
        input.arr_name + "_" + "_".join(map(str, idx))
        for idx in np.ndindex(input.curr_shape)
    ]
    outputs = [
        output.arr_name + "_" + "_".join(map(str, idx))
        for idx in np.ndindex(output.curr_shape)
    ]
    print(
        f"define void @forward({', '.join([f'float %{a}' for a in inputs] + [f'float* %{a}' for a in outputs])}) {{\n",
        file=FILE,
    )
    # for _arg_name, arg in Args.items():
    #     if isinstance(arg, ArrayDecl):
    #         typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
    #         # shape = arg.curr_shape if len(arg.curr_shape) > 1 else (arg.curr_shape,)
    #         for idx in itertools.product(*[range(a) for a in arg.curr_shape]):
    #             idx = list(map(str, idx))
    #             print(
    #                 f"%{arg.var_name}_{'_'.join(idx)}_gep = getelementptr {typ}, {typ}* %{arg.var_name}, i32 0, {', '.join([f'i32 {i}' for i in idx])}", file=FILE
    #                 # f"%mac_{mac.id}_{i}_gep = getelementptr {typ}, {typ}* %mac_{mac.id}, i16 0, i16 {i}", file=OLD_FILE
    #             )
    #             if arg.input:
    #                 print(
    #                     # f"%glob_{glo.var_name}_idx_{'_'.join(idx)} = load float, float* getelementptr inbounds ({typ}, {typ}* %glob_{glo.var_name}, i64 0, {', '.join([f'i64 {i}' for i in idx])})"
    #                     f"%{arg.var_name}_{'_'.join(idx)} = load float, float* %{arg.var_name}_{'_'.join(idx)}_gep", file=FILE
    #                 )

    OLD_FILE = FILE
    FILE = io.StringIO()
    forward()

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for idx, reg in output.registers.items():
        id = "_".join(map(str, idx))
        print(f"store float {reg}, float* %{output.arr_name}_{id}", file=OLD_FILE)

    print("ret void", file=OLD_FILE)
    print("}", file=OLD_FILE)


def Forward(forward, max_range):
    Args = get_default_args(forward)
    LLVMForward(Args["_arg0"], Args["_arg1"], forward, [], max_range=max_range)


def get_ssas_from_ir_line(line):
    line = re.sub(r", align \d+", "", line)
    idents = [f for f, _ in re.findall(r"(%[\d|a-z|_]*|([0-9]*[.])+[0-9]+)", line)]
    if not idents:
        print(line)
        return None, None

    if (
        "fmul" in line
        or "fadd" in line
        or "fcmp" in line
        or "fsub" in line
        or "fdiv" in line
    ):
        assign, *deps = idents
    elif "store" in line:
        dep, assign = idents
        deps = [dep]
    elif "expf" in line or "relu" in line:
        assign, *deps = idents
    elif "fptrunc" in line:
        assign, *deps = idents
    elif "define void @forward" in line:
        inputs = [
            f.replace("float ", "")
            for f, _ in re.findall(r"(float (%[\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace("float* ", "")
            for f, _ in re.findall(r"(float\* (%[\d|a-z|_]*))", line)
        ]
        return inputs, outputs
    else:
        raise Exception(line)

    return assign, deps


def topological_sort_grouped(G):
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    while zero_indegree:
        yield zero_indegree
        new_zero_indegree = []
        for v in zero_indegree:
            for _, child in G.edges(v):
                indegree_map[child] -= 1
                if not indegree_map[child]:
                    new_zero_indegree.append(child)
        zero_indegree = new_zero_indegree


def crawl_graph(fp):
    lines = open(fp).readlines()
    G = nx.DiGraph()
    inputs = None
    outputs = None
    for line in lines:
        assign, deps = get_ssas_from_ir_line(line)
        if "define" in line:
            inputs = assign
            outputs = deps
            for assig in assign:
                G.add_node(assig)
            for dep in deps:
                G.add_node(dep)
        else:
            if assign is not None:
                if assign not in G.nodes:
                    G.add_node(assign)
                for dep in deps:
                    if dep not in G.nodes:
                        G.add_node(dep)
                    G.add_edge(dep, assign)
    for stage in topological_sort_grouped(G):
        print(len(stage))
    # for inp in inputs:
    #     for outp in outputs:
    #         try:
    #             paths = nx.all_simple_paths(G, inp, outp)
    #             for path in paths:
    #                 print("path", inp, outp, len(path))
    #         except nx.exception.NetworkXNoPath:
    #             print("no path", inp, outp)
    #             continue


if __name__ == "__main__":
    fp = sys.argv[1]
    crawl_graph(fp)

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
