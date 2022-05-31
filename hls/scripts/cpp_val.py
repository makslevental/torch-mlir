import io
import itertools
import json
import os
import re
from collections import defaultdict
from textwrap import dedent

import networkx as nx

from hls.scripts.mlir_ops import get_default_args, ArrayIndex, index_map


def format_cst(cst):
    return cst


VAR_COUNT = 0

FILE = None


class CPPVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.var_id = f"val_{VAR_COUNT}"
        VAR_COUNT += 1

    def __mul__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(* ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"float {v} = fmul(1.0, {other});", file=FILE)
        elif "-1" in f"{other}":
            print(f"float {v} = fmul({self}, 1.0);", file=FILE)
        else:
            print(f"float {v} = fmul({self}, {other});", file=FILE)
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(+ ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"float {v} = fadd(0.0, {other});", file=FILE)
        elif "-1" in f"{other}":
            print(f"float {v} = fadd({self}, 0.0);", file=FILE)
        else:
            print(f"float {v} = fadd({self}, {other});", file=FILE)
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(- ({self}) ({other}))")
        print(f"float {v} = fsub({self}, {other});", file=FILE)
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(/ ({self}) ({other}))")
        print(f"float {v} = fdiv({self}, {other});", file=FILE)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(> ({self}) ({other}))")
        print(f"float {v} = fcmpugt({self}, {other});", file=FILE)
        return v

    def __str__(self):
        return f"var_{self.var_id}"


class CPPConstant(CPPVal):
    def __init__(self, name):
        super(CPPConstant, self).__init__(name)
        self._fmt = f"{format_cst(self.name)}"

    def __str__(self):
        return self._fmt


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
        self.index_to_vals = defaultdict(list)

    def __setitem__(self, index, value):
        index = self.idx_map(index)
        assert not self.input
        self.registers[index] = value

    def __getitem__(self, index: ArrayIndex):
        index = self.idx_map(index)
        if index not in self.registers:
            v = ArrayVal(f"{self.arr_name}", index, self)
            self.registers[index] = v

        v = self.registers[index]
        return v

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class ArrayVal(CPPVal):
    array: ArrayDecl

    def __init__(self, name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(name)
        self.array = array
        self.var_id = "_".join(map(str, val_id))

    def __str__(self):
        return f"var{self.name}_{self.var_id}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"glob{self.name}_{self.var_id}"


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    a, b, c = inps
    v = CPPVal(f"(fmuladd {a} {b} {c})")
    print(f"float {v} = fmuladd({a}, {b}, {c});", file=FILE)
    return v


def FMac(a, b, arr, idx):
    inps = [a, b, arr[idx]]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    a, b, c = inps
    print(f"float {c} = fmuladd({a}, {b}, {c});", file=FILE)
    return c


def Add(a, arr, idx):
    inps = [a, arr[idx]]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    a, b = inps
    print(f"float {b} = fadd({a}, {b});", file=FILE)
    return b


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


def make_args_globals(Args):
    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                if arg.input:
                    args.append(f"float var{arg.arr_name}_{id}")
                elif arg.output:
                    args.append(f"float* var{arg.arr_name}_{id}")
        elif isinstance(arg, GlobalArray):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                globals.append(f"float glob{arg.arr_name}_{id}")

    return args, globals


def make_op_json(op_name, id):
    return {
        "c_function_name": f"f{op_name}_{id}",
        "rtl_top_module_name": f"f{op_name}_{id}",
        "c_files": [{"c_file": f"jsons/f{op_name}_{id}.cpp", "cflag": ""}],
        "rtl_files": [f"jsons/f{op_name}_{id}.v"],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {"data_read_in": "a1"},
            },
            {
                "c_name": "b1",
                "c_port_direction": "in",
                "rtl_ports": {"data_read_in": "b1"},
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {"data_write_out": "z1", "data_write_valid": "z1_ap_vld"},
            },
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue",
        },
        "rtl_performance": {"latency": "1", "II": "1"},
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1",
        },
    }


def make_single_arg_op_json(op_name, id):
    return {
        "c_function_name": f"f{op_name}_{id}",
        "rtl_top_module_name": f"f{op_name}_{id}",
        "c_files": [{"c_file": f"jsons/f{op_name}_{id}.cpp", "cflag": ""}],
        "rtl_files": [f"jsons/f{op_name}_{id}.v"],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {"data_read_in": "a1"},
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {"data_write_out": "z1", "data_write_valid": "z1_ap_vld"},
            },
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue",
        },
        "rtl_performance": {"latency": "1", "II": "1"},
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1",
        },
    }


def make_CPP_2_arg_op_prototype(op_name, id):
    return f"""
define void @_Z12f{op_name}_{id}ffPf(float %a1, float %b1, float* %z1) #0 {{
entry:
  call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
  call void (...) @_ssdm_op_BlackBox(float %a1, float %b1, float* %z1)
  call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
  ret void
}}
"""


def make_CPP_1_arg_op_prototype(op_name, id):
    return f"""
define void @_Z12f{op_name}_{id}ffPf(float %a1, float* %z1) #0 {{
entry:
  call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
  call void (...) @_ssdm_op_BlackBox(float %a1, float* %z1)
  call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
  ret void
}}
"""


def make_CPP_prefix():
    return """
    
; ModuleID = 'forward.cpp'
source_filename = "forward.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@0 = private unnamed_addr constant [10 x i8] c"ap_memory\\00"
@1 = private unnamed_addr constant [1 x i8] zeroinitializer
@2 = private unnamed_addr constant [8 x i8] c"forward\\00"

declare void @_ssdm_op_BlackBox(...)

declare void @_ssdm_op_SpecIPCore(...)

declare void @_ssdm_op_SpecInterface(...)

declare void @_ssdm_op_SpecBitsMap(...)

declare void @_ssdm_InlineSelf(...)

declare void @_ssdm_op_SpecTopModule(...)
"""


def make_blackbox():
    os.makedirs("jsons", exist_ok=True)
    for op_name, op_dict in [
        ("mul", MULS),
        ("add", ADDS),
        ("div", DIVS),
        ("sub", SUBS),
        ("ugt", UGTS),
    ]:
        template = open(f"f{op_name}.v").read()
        for idx, op in op_dict.items():
            s = f"void f{op_name}_{op.id}(float a1, float b1, float* z1)"
            print(f"{s};", file=OLD_FILE)

            fmul_cpp = open(f"jsons/f{op_name}_{op.id}.cpp", "w")
            fmul_v = open(f"jsons/f{op_name}_{op.id}.v", "w")
            fmul_cpp.write(f"{s} {{}}\n")
            json.dump(
                make_op_json(op_name, op.id),
                open(f"jsons/f{op_name}_{op.id}.json", "w"),
            )
            fmul_v.write(template.replace("XXX", op.id))

    for op_name, op_dict in [("relu", ReLUS), ("exp", EXPS)]:
        template = open(f"f{op_name}.v").read()
        for idx, op in op_dict.items():
            s = f"void f{op_name}_{op.id}(float a1, float* z1)"
            print(f"{s};", file=OLD_FILE)

            fmul_cpp = open(f"jsons/f{op_name}_{op.id}.cpp", "w")
            fmul_v = open(f"jsons/f{op_name}_{op.id}.v", "w")
            fmul_cpp.write(f"{s} {{}}\n")
            json.dump(
                make_single_arg_op_json(op_name, op.id),
                open(f"jsons/f{op_name}_{op.id}.json", "w"),
            )
            fmul_v.write(template.replace("XXX", op.id))


def make_ops():
    return dedent(
        """\
    float fmul(float a, float b) {
        return a * b;
    } 
    float fdiv(float a, float b) {
        return a / b;
    } 
    float relu(float a) {
        return a;
    } 
    """
    )


def CPPForward(Args, output, forward):
    global FILE
    FILE = open("forward.cpp", "w")

    args, globals = make_args_globals(Args)

    OLD_FILE = FILE
    FILE = io.StringIO()
    print('extern "C" ' + f"void forward({', '.join(args + globals)}) {{\n", file=FILE)

    forward()

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for index, value in output.registers.items():
        id = "_".join(map(str, index))
        print(f"*var_{output.arr_name}_{id} = {value};", file=OLD_FILE)

    print("return;", file=OLD_FILE)
    print("}", file=OLD_FILE)

    OLD_FILE.close()


def Forward(forward, max_range, worker_id=None):
    Args = get_default_args(forward)
    CPPForward(Args, Args["_arg1"], forward)


def get_ssas_from_ir_line(line):
    line = re.sub(r", align \d+", "", line)
    idents = [f for f, _ in re.findall(r"(var_[\d|a-z|_]*|([0-9]*[.])+[0-9]+)", line)]
    if not idents or "declare" in line or "//" in line:
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
            try:
                float(d)
            except:
                deps.append(d)
        if "fmuladd" in line:
            op = "fmuladd"
        else:
            op = line.split("=")[1].strip().split("(")[0]
    elif "*" in line and "=" in line:
        assign, dep = idents
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
    elif "forward" in line:
        inputs = [
            f.replace("float ", "")
            for f, _ in re.findall(r"(float (var_[\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace("float* ", "")
            for f, _ in re.findall(r"(float\* (var_[\d|a-z|_]*))", line)
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
    G = nx.MultiDiGraph()

    for line in lines:
        assign, deps, op = get_ssas_from_ir_line(line)
        if "forward" in line:
            for assig in assign:
                G.add_node(assig, op="input")
            for dep in deps:
                G.add_node(dep, op="output")
        else:
            if assign is not None:
                if assign not in G.nodes:
                    G.add_node(assign, op=op)
                for i, dep in enumerate(deps):
                    if dep not in G.nodes:
                        assert "constant" in dep, dep
                        G.add_node(dep, op="constant")
                    G.add_edge(dep, assign, pos=i, op=op)
    design = {
        "G": nx.json_graph.node_link_data(G),
        "topo_sort": [],
    }
    for i, stage in enumerate(topological_sort_grouped(G)):
        design["topo_sort"].append(stage)

    fp_dir = os.path.split(fp)[0]
    json.dump(design, open(f"{fp_dir}/design.json", "w"), indent=2)


if __name__ == "__main__":
    fp = "/Users/mlevental/dev_projects/torch-mlir/hls/examples/Linear.1/forward.cpp"
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
