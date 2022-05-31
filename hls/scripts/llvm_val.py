from __future__ import annotations

import io
import itertools
import os
import re
import struct
import sys
from collections import defaultdict, deque
from textwrap import dedent
from typing import Tuple

import networkx as nx
import numpy as np
from llvmlite.ir import Constant, FloatType

from hls.scripts.mlir_ops import get_default_args, get_array_type, index_map


def double_to_hex(f):
    return hex(struct.unpack("<Q", struct.pack("<d", f))[0])


def float_to_hex(f):
    return hex(struct.unpack("<H", struct.pack("<e", f))[0])


def format_cst(cst):
    c = Constant(FloatType(), float(cst))
    # return double_to_hex(float(cst))
    # return float(np.random.randint(1, 100000))
    return str(c).replace("double ", "").replace("half ", "")
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


def np_to_hex(x):
    return hex(np.float16(x).view("H"))[2:].zfill(16)


VAR_COUNT = 0

FILE = None
# DTYPE = "float"
DTYPE = "half"


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
            print(f"{v} = fmul {DTYPE} 1.0, {other}", file=FILE)
        elif "-1" in f"{other}":
            print(f"{v} = fmul {DTYPE} {self}, 1.0", file=FILE)
        else:
            print(f"{v} = fmul {DTYPE} {self}, {other}", file=FILE)
        # print(
        #     f"{v} = call {DTYPE} @llvm.fmuladd.f32({DTYPE} {self}, {DTYPE} {other}, {DTYPE} 0.0)"
        # )
        return v

    def __add__(self, other):
        # <result> = fadd {DTYPE} 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(+ ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"{v} = fadd {DTYPE} 0.0, {other}", file=FILE)
        elif "-1" in f"{other}":
            print(f"{v} = fadd {DTYPE} {self}, 0.0", file=FILE)
        else:
            print(f"{v} = fadd {DTYPE} {self}, {other}", file=FILE)
        # print(
        #     f"{v} = call {DTYPE} @llvm.fmuladd.f32({DTYPE} {self}, {DTYPE} 1.0, {DTYPE} {other})"
        # )
        return v

    def __sub__(self, other):
        # <result> = fsub {DTYPE} 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(- ({self}) ({other}))")
        print(f"{v} = fsub {DTYPE} {self}, {other}", file=FILE)
        return v

    def __truediv__(self, other):
        # <result> = fdiv {DTYPE} 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(/ ({self}) ({other}))")
        print(f"{v} = fdiv {DTYPE} {self}, {other}", file=FILE)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt {DTYPE} 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = LLVMConstant(other)
        v = LLVMVal(f"(> ({self}) ({other}))")
        print(f"{v} = fcmp ugt {DTYPE} {self}, {other}", file=FILE)
        return v

    def __neg__(self):
        v = LLVMVal(f"(- ({self}))")
        print(f"{v} = fneg {DTYPE} {self}", file=FILE)
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


def replace_trailing_num(global_name):
    reps = ["a", "b", "c", "d", "e", "f"]
    for i, rep in enumerate(reps):
        if f"f32_{i}" in global_name:
            return global_name.replace(f"f32_{i}", f"f32_{rep}")
    return global_name


class GlobalArray:
    def __init__(self, name, global_name, global_array):
        self.name = name
        self.arr_name = replace_trailing_num(global_name)
        self.global_array = global_array
        self.curr_shape = global_array.shape
        self.csts = {}

    def __getitem__(self, index: ArrayIndex):
        if index not in self.csts:
            v = GlobalArrayVal(self.arr_name, index, self)
            self.csts[index] = v
            # print(
            #     f"{v} = fptrunc double {format_cst(self.global_array[index].view('float64'))} to {DTYPE}",
            #     file=FILE,
            # )
        return self.csts[index]


UNIQS = set()
ALL = []


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
            elif WORKER_ID is not None and index != WORKER_IDXS[WORKER_ID]:
                v = ArrayVal(f"other_worker_idx", index, self)
                if index not in UNIQS:
                    UNIQS.add(index)
                    ALL.append([list(index), self.arr_name])
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
    exp_name = "exp"
    print(f"{v} = alloca {DTYPE}, align 4", file=FILE)
    print(f"call void @_Z12{exp_name}fPf({DTYPE} {val}, {DTYPE}* {v})", file=FILE)
    vv = LLVMVal(f"(load ({v})")
    print(f"{vv} = load {DTYPE}, {DTYPE}* {v}, align 4", file=FILE)
    return vv


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
                f"{v} = call {DTYPE} @llvm.fmuladd.f32({DTYPE} {a}, {DTYPE} {b}, {DTYPE} {c})",
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
        _idx[0: len(idx)] = idx
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
        # exp_name = "relu"
        # print(f"{v} = alloca {DTYPE}, align 4", file=FILE)
        # print(f"call void @_Z12{exp_name}fPf({DTYPE} {val}, {DTYPE}* {v})", file=FILE)
        # vv = LLVMVal(f"(load ({v})")
        # print(f"{vv} = load {DTYPE}, {DTYPE}* {v}, align 4", file=FILE)
        print(f"{v} = fsub {DTYPE} {val}, 0.0", file=FILE)
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
    if WORKER_ID is not None:
        if WORKER_ID < len(task_idxs):
            task_idx = task_idxs[WORKER_ID]
            body(*task_idx)
        else:
            # TODO: how many no-ops though?
            print("call void @llvm.donothing()", file=FILE)
    else:
        for idx in task_idxs:
            body(*idx)


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = LLVMConstant(v)
    v = LLVMVal(f"(fmuladd ({a}) ({b}) ({c})")
    print(
        f"{v} = call {DTYPE} @llvm.fmuladd.f32({DTYPE} {inps[0]}, {DTYPE} {inps[1]}, {DTYPE} {inps[2]})",
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


def make_geps_from_arr(arg):
    typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
    insts = []
    for idx in itertools.product(*[range(a) for a in arg.curr_shape]):
        idx = list(map(str, idx))
        insts.extend(
            [
                f"%{arg.arr_name}_{'_'.join(idx)}_gep = getelementptr {typ}, {typ}* %{arg.arr_name}, i32 0, {', '.join([f'i32 {i}' for i in idx])}",
                # f"%mac_{mac.id}_{i}_gep = getelementptr {typ}, {typ}* %mac_{mac.id}, i16 0, i16 {i}", file=OLD_FILE
                # f"%glob_{glo.arr_name}_idx_{'_'.join(idx)} = load {DTYPE}, {DTYPE}* getelementptr inbounds ({typ}, {typ}* %glob_{glo.arr_name}, i64 0, {', '.join([f'i64 {i}' for i in idx])})"
                f"%{arg.arr_name}_{'_'.join(idx)} = load {DTYPE}, {DTYPE}* %{arg.arr_name}_{'_'.join(idx)}_gep",
            ]
        )
    return insts


def make_arr_args(arg):
    typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
    return f"{typ}* noalias %{arg.arr_name}"
    # inputs = [
    #     input.arr_name + "_" + "_".join(map(str, idx))
    #     for idx in np.ndindex(input.curr_shape)
    # ]


def make_blackbox_ir():
    return dedent(
        f"""\
    
    @0 = private unnamed_addr constant [10 x i8] c"ap_memory\\00"
    @1 = private unnamed_addr constant [1 x i8] zeroinitializer
    @2 = private unnamed_addr constant [8 x i8] c"forward\\00"
    
    declare void @_ssdm_op_BlackBox(...)

    declare void @_ssdm_op_SpecIPCore(...)

    declare void @_ssdm_op_SpecInterface(...)

    declare void @_ssdm_op_SpecBitsMap(...)

    declare void @_ssdm_InlineSelf(...)

    declare void @_ssdm_op_SpecTopModule(...)
    
    define void @_Z12relufPf({DTYPE} %a1, {DTYPE}* %z1) {{
        entry:
      call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
      call void (...) @_ssdm_op_BlackBox({DTYPE} %a1, {DTYPE}* %z1)
      call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
      ret void
    }}
    
    define void @_Z12expfPf({DTYPE} %a1, {DTYPE}* %z1) {{
        entry:
      call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
      call void (...) @_ssdm_op_BlackBox({DTYPE} %a1, {DTYPE}* %z1)
      call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
      ret void
    }}

    """
    )


def LLVMForward(Args, output, forward, max_range):
    global FILE, WORKER_IDXS
    WORKER_IDXS = list(np.ndindex(*max_range))
    os.makedirs("workers", exist_ok=True)
    FILE = open(f"workers/forward_{WORKER_ID if WORKER_ID is not None else ''}.ll", "w")
    print('source_filename = "LLVMDialectModule"', file=FILE)

    # print(make_blackbox_ir(), file=FILE)

    inputs = [make_arr_args(Args["_arg0"])] + [
        make_arr_args(a) for a in Args.values() if isinstance(a, GlobalArray)
    ]
    outputs = [make_arr_args(Args["_arg1"])]
    print("declare half @llvm.fmuladd.f32(half %a, half %b, half %c)", file=FILE)
    print(f"define void @forward({', '.join(inputs + outputs)}) {{\n", file=FILE)

    for arg in Args.values():
        for inst in make_geps_from_arr(arg):
            print(inst, file=FILE)

    OLD_FILE = FILE
    FILE = io.StringIO()
    forward()

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for idx, reg in output.registers.items():
        id = "_".join(map(str, idx))
        print(
            f"store {DTYPE} {reg}, {DTYPE}* %{output.arr_name}_{id}_gep", file=OLD_FILE
        )

    print("ret void", file=OLD_FILE)
    print("}", file=OLD_FILE)


def Forward(forward, max_range, worker_id=None):
    global WORKER_ID
    WORKER_ID = worker_id
    Args = get_default_args(forward)
    LLVMForward(Args, Args["_arg1"], forward, max_range=max_range)


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
            f.replace(f"{DTYPE} ", "")
            for f, _ in re.findall(rf"({DTYPE} (%[\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace(f"{DTYPE}* ", "")
            for f, _ in re.findall(rf"({DTYPE}\* (%[\d|a-z|_]*))", line)
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
#   call void @_Z12fmul_0_8_0_6ffPf({DTYPE} %v123110, {DTYPE} 6.000000e+00, {DTYPE}* %v5891)
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
