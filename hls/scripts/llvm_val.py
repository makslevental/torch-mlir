from __future__ import annotations

import io
import itertools
import os
import re
import struct
from collections import defaultdict, deque
import sys
from typing import Tuple
from pprint import pprint
from llvmlite.ir import (
    Constant,
    FloatType,
)

import networkx as nx
import numpy as np
from hls.scripts.mlir_ops import get_default_args, get_array_type, index_map


def double_to_hex(f):
    return hex(struct.unpack("<Q", struct.pack("<d", f))[0])


def float_to_hex(f):
    return hex(struct.unpack("<H", struct.pack("<e", f))[0])


def format_cst(cst):
    c = Constant(FloatType(), float(cst))
    # return double_to_hex(float(cst))
    # return float(np.random.randint(1, 100000))
    return str(c).replace("double ", "").replace("float ", "")
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


def np_to_hex(x):
    return hex(np.float16(x).view("H"))[2:].zfill(16)


VAR_COUNT = 0

FILE = None


class LLVMVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.var_id = f"val_{VAR_COUNT:03}"
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
            print(
                f"{v} = fptrunc double {format_cst(self.global_array[index].view('float64'))} to float",
                file=FILE,
            )
        return self.csts[index]

UNIQS = []

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
        index = self.idx_map(index)
        if index not in self.registers:
            if self.input:
                v = ArrayVal(f"{self.arr_name}", index, self)
            elif index != WORKER_IDXS[WORKER_ID]:
                v = ArrayVal(f"other_worker_idx", index, self)
                if index not in UNIQS:
                    UNIQS.append(list(index))
            else:
                v = LLVMConstant("-666")
            self.registers[index] = v

        v = self.registers[index]
        return v

    def __setitem__(self, index, value):
        index = self.idx_map(index)
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


WORKER_ID = None
WORKER_IDXS = None
NUM_OPS_BODY = {}

def ParFor(body, ranges):
    task_idxs = sorted(list(itertools.product(*ranges)))
    if WORKER_ID < len(task_idxs):
        task_idx = task_idxs[WORKER_ID]
        body(*task_idx)
    else:
        # TODO: how many no-ops though?
        print("call void @llvm.donothing()", file=FILE)


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = LLVMConstant(v)
    v = LLVMVal(f"(fmuladd ({a}) ({b}) ({c})")
    print(
        f"{v} = call float @llvm.fmuladd.f32(float {inps[0]}, float {inps[1]}, float {inps[2]})",
        file=FILE,
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
    global FILE, WORKER_IDXS
    WORKER_IDXS = list(np.ndindex(*max_range))
    os.makedirs("workers", exist_ok=True)
    FILE = open(f"workers/forward_{WORKER_ID}.ll", "w")
    print('source_filename = "LLVMDialectModule"', file=FILE)
    print("declare float @expf(float)", file=FILE)
    print("declare void @llvm.donothing() nounwind readnone", file=FILE)
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


def Forward(forward, max_range, worker_id=0):
    global WORKER_ID
    WORKER_ID = worker_id
    Args = get_default_args(forward)
    LLVMForward(Args["_arg0"], Args["_arg1"], forward, [], max_range=max_range)
    pprint(UNIQS)


def get_ssas_from_ir_line(line):
    line = re.sub(r", align \d+", "", line)
    idents = [f for f, _ in re.findall(r"(%[\d|a-z|_]*|([0-9]*[.])+[0-9]+)", line)]
    if not idents or "declare" in line:
        return None, None, None

    if (
        "fmul" in line
        or "fadd" in line
        or "fcmp" in line
        or "fsub" in line
        or "fdiv" in line
        or "fmuladd" in line
    ):
        assign, *_deps = idents
        deps = []
        for d in _deps:
            # deps.append(d)
            try:
                float(d)
            except:
                deps.append(d)
        if "fmuladd" in line:
            op = "fmuladd"
        else:
            op = line.split("=")[1].strip().split(" ")[0]
    elif "store" in line:
        dep, assign = idents
        deps = [dep]
        op = "store"
    elif "expf" in line:
        assign, *deps = idents
        op = "expf"
    elif "relu" in line:
        assign, *deps = idents
        op = "relu"
    elif "fptrunc" in line:
        assign, *_deps = idents
        deps = _deps
        op = "constant"
    elif "define void @forward" in line:
        inputs = [
            f.replace("float ", "")
            for f, _ in re.findall(r"(float (%[\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace("float* ", "")
            for f, _ in re.findall(r"(float\* (%[\d|a-z|_]*))", line)
        ]
        return inputs, outputs, ""
    else:
        raise Exception(line)

    return assign, deps, op


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
    lines = open(fp, "r").readlines()
    G = nx.DiGraph()
    inputs = None
    outputs = None
    for line in lines:
        assign, deps, op = get_ssas_from_ir_line(line)
        if "define" in line:
            inputs = assign
            outputs = deps
            for assig in assign:
                G.add_node(assig, op="input")
            for dep in deps:
                G.add_node(dep, op="output")
        else:
            if assign is not None:
                if assign not in G.nodes:
                    G.add_node(assign, op=op)
                for dep in deps:
                    if dep not in G.nodes:
                        assert "other_worker" in dep
                        G.add_node(dep, op="other_worker")
                    G.add_edge(dep, assign)
    for i, stage in enumerate(topological_sort_grouped(G)):
        print(i, len(stage))
        print(sorted([(s, G.nodes[s]["op"]) for s in stage]))
        for s in stage:
            preds = list(G.predecessors(s))
            if preds:
                print(s, preds)
    print()
    for inp in inputs:
        for outp in outputs:
            try:
                path = nx.shortest_path(G, inp, outp)
                print("path", inp, outp, len(path))
            except nx.exception.NetworkXNoPath:
                # print("no path", inp, outp)
                continue


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
# #pragma HLS BIND_STORAGE variable=v87 type=RAM_2P impl=BRAM latency=2V
