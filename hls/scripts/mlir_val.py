import argparse
import ast
import io
import itertools
import json
import os
import re
from textwrap import dedent

import networkx as nx

from hls.scripts.mlir_ops import get_default_args, ArrayIndex, index_map


def format_cst(cst):
    return int(float(cst))
    # c = Constant(FloatType(), float(cst))
    # return str(c).replace("double ", "").replace("half ", "")


VAR_COUNT = 0
OP_COUNT = 0
OP_COUNT_ZFILL = 5
FILE = None
OP_LAYER_PREFIX = "layer0"
DTYPE = "f32"

LAST_IDX_TO_OP_ID = {}
VAL_TO_IDX = {}
DEPS = set()


def make_unregistered_op(op, *args):
    # return f"\"{op}_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}\"({', '.join(map(str, args))}) {{ opr = \"{op}\" }} : ({', '.join([DTYPE for _ in args])}) -> {DTYPE}"
    return f""""{op}"({', '.join(map(str, args))}) {{ opr = "{op}", pe ="{IDX}" }} : ({', '.join([DTYPE for _ in args])}) -> {DTYPE}"""


def print_op(op, v, *args):
    global OP_COUNT
    if IDX is not None:
        if (op, IDX) in LAST_IDX_TO_OP_ID:
            DEPS.add((LAST_IDX_TO_OP_ID[op, IDX], OP_COUNT))
        LAST_IDX_TO_OP_ID[op, IDX] = OP_COUNT
        VAL_TO_IDX[v] = IDX
    if "arith" in op:
        print(f"{v} = arith.constant {args[0]} : {DTYPE}", file=FILE)
    else:
        print(f"{v} = {make_unregistered_op(op, *args)}", file=FILE)
    OP_COUNT += 1


class MLIRVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.val_id = f"val_{VAR_COUNT}"
        VAR_COUNT += 1

    def __mul__(self, other):
        if isinstance(other, (float, int, bool)):
            other = MLIRConstant(other)
        v = MLIRVal(f"(* ({self}) ({other}))")
        print_op("fmul", v, self, other)

        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = MLIRConstant(other)
        v = MLIRVal(f"(+ ({self}) ({other}))")
        print_op("fadd", v, self, other)
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = MLIRConstant(other)
        v = MLIRVal(f"(- ({self}) ({other}))")
        print_op("fsub", v, self, other)
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = MLIRConstant(other)
        v = MLIRVal(f"(/ ({self}) ({other}))")
        print_op("fdiv", v, self, other)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = MLIRConstant(other)
        v = MLIRVal(f"(> ({self}) ({other}))")
        print_op("fcmpugt", v, self, other)
        return v

    def __neg__(self):
        # <result> = fcmp ugt float 4.0, 5.0
        v = MLIRVal(f"(- {self})")
        print_op("fneg", v, self)
        return v

    def __str__(self):
        return f"%{self.val_id}"


EMITTED_CSTS = set()


class MLIRConstant(MLIRVal):
    def __init__(self, name):
        super(MLIRConstant, self).__init__(name)
        self.fmted_cst = f"{format_cst(self.name)}"
        if str(self) not in EMITTED_CSTS:
            print_op("arith.constant", self, name)
            EMITTED_CSTS.add(str(self))

    def __str__(self):
        return f"%cst{self.fmted_cst}"


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

    def __setitem__(self, index, value):
        if isinstance(value, (float, int, bool)):
            value = MLIRConstant(value)
        index = self.idx_map(index)
        assert not self.input
        self.registers[index] = value

    def __getitem__(self, index: ArrayIndex):
        index = self.idx_map(index)
        if index not in self.registers:
            self.registers[index] = self.make_val(index)
        v = self.registers[index]
        return v

    def make_val(self, index):
        name = self.arr_name
        if self.output:
            name = f"output_{name}"
        v = ArrayVal(name, index, self)
        return v

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class ArrayVal(MLIRVal):
    array: ArrayDecl

    def __init__(self, arr_name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(arr_name)
        self.array = array
        self.val_id = "_".join(map(str, val_id))

    def __str__(self):
        return f"%{self.name}_{self.val_id}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"%{self.name}_{self.val_id}"


class FMAC:
    def __init__(self, *pe_idx):
        global IDX
        if len(pe_idx) < 5:
            _idx = 5 * [0]
            _idx[-len(pe_idx) :] = pe_idx
            pe_idx = tuple(_idx)
        self.pe_idx = IDX = pe_idx
        print(f"// pe {pe_idx} starts", file=FILE)
        self.most_recent_res = None

    def Add(self, a, b):
        self.most_recent_res = a + b
        return self.most_recent_res

    def Mul(self, a, b):
        self.most_recent_res = a * b
        return self.most_recent_res

    def Result(self):
        print(f"// pe {self.pe_idx} ends", file=FILE)
        return self.most_recent_res


def Add(a, arr, idx):
    if isinstance(a, (float, int, bool)):
        a = MLIRConstant(a)

    # handle the first time you process a constant of input
    b = bb = arr[idx]
    if isinstance(b, (GlobalArrayVal, MLIRConstant)):
        bb = arr[idx] = arr.make_val(idx)
    print_op("fadd", bb, a, b)
    return bb


def ReduceAdd(src_arr: ArrayDecl, dst_arr: ArrayDecl):
    global IDX
    prev_sums = list(src_arr.registers.values())
    while len(prev_sums) > 1:
        next_sums = []
        while len(prev_sums) > 1:
            left = prev_sums.pop()
            IDX = VAL_TO_IDX[left]
            next_sums.append(left + prev_sums.pop())
        if len(prev_sums) == 1:
            left = next_sums[-1]
            IDX = VAL_TO_IDX[left]
            next_sums[-1] = left + prev_sums[0]
        prev_sums = next_sums
    dst_arr[
        0,
    ] = prev_sums[0]


def Copy(dst: ArrayDecl, src: ArrayDecl):
    assert isinstance(src, ArrayDecl)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


def ReLU(x):
    v = MLIRVal(f"(relu {x})")
    print_op("relu", v, x)
    return v


def make_args_globals(Args):
    args = []
    outputs = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                if arg.input:
                    args.append(f"%{arg.arr_name}_{id}: {DTYPE}")
                elif arg.output:
                    outputs.append(f"{DTYPE}")
        elif isinstance(arg, GlobalArray):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                globals.append(f"%{arg.arr_name}_{id}: {DTYPE}")

    return args, outputs, globals


LATENCIES = {
    "fadd": 8,
    "fmul": 3,
    "fmuladd": 6,
    "relu": 1,
    "neg": 1,
    "fneg": 1,
    "constant": 1,
    "copy": 1,
}


def make_latency_attrs(file):
    deps = sorted([list(dep) for dep in DEPS])
    operator_types = [
        f"""{{ name = "{op}", latency = {lat}, limit = 2592 }}"""
        for op, lat in LATENCIES.items()
    ]
    print(
        f"""
 attributes {{
  auxdeps = {deps},
  operatortypes = [
    {', '.join(operator_types)}
    ] }} \n{{
""",
        file=file,
    )


def MLIRForward(Args, output, forward):
    global FILE
    fn = os.environ.get("FN", "regular")
    FILE = open(f"forward{f'_{fn}' if fn else ''}.mlir", "w")

    args, outputs, globals = make_args_globals(Args)

    OLD_FILE = FILE
    print(
        f"func.func @forward({', '.join(args + globals)}) -> ({', '.join(outputs)})\n",
        file=OLD_FILE,
    )
    FILE = io.StringIO()
    forward()
    make_latency_attrs(OLD_FILE)
    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    rets = []
    for index, value in output.registers.items():
        rets.append(str(value))

    print(
        f"return {', '.join(rets)}: {', '.join([DTYPE for _ in rets])}", file=OLD_FILE
    )
    print("}", file=OLD_FILE)

    OLD_FILE.close()


def Forward(forward, max_range=None, worker_id=None):
    Args = get_default_args(forward)
    MLIRForward(Args, Args["_arg1"], forward)


reg_idents = re.compile(r"(%[\d|a-z|_]*|([0-9]*[.])+[0-9]+)")
reg_start_time = re.compile(r"lpStartTime = (\d+)")
reg_opr = re.compile(r'opr = "([a-z]+)"')
reg_pe = re.compile(r'pe = "(.*)"')


def get_ssas_from_ir_line(line):
    start_time = reg_start_time.search(line)
    start_time = start_time.groups()[0] if start_time else start_time
    if "arith.constant" in line:
        opr = "constant"
    else:
        opr = reg_opr.search(line)
        opr = opr.groups()[0] if opr else opr
    pe_idx = reg_pe.search(line)
    pe_idx = ast.literal_eval(pe_idx.groups()[0]) if pe_idx else pe_idx

    return int(start_time) + 1, opr, pe_idx


def build_regular_code_graph(fp):
    lines = open(fp, "r").readlines()
    G = nx.MultiDiGraph()

    for line in lines:
        if "attributes" in line:
            line = line.split("attributes")[0].strip()
        idents = [i.replace("%", "v") for i, _ in reg_idents.findall(line)]
        if not idents:
            continue
        if "forward" in line:
            for i, ident in enumerate(idents):
                G.add_node(ident, op="arg", arg_pos=i)
            continue

        try:
            start_time, op, pe_idx = get_ssas_from_ir_line(line)
        except:
            print(line)
            raise

        assign, *deps = idents
        if "return" in line:
            G.add_node("return", op="return", start_time=start_time)
            for i, ident in enumerate(idents):
                G.add_edge(ident, "return", pos=i, op="return")
            continue

        if op == "constant":
            G.add_node(assign, op=op, literal=float(deps[0]))
            continue

        assert pe_idx is not None or op == "copy"
        if assign is not None:
            if assign not in G.nodes:
                G.add_node(
                    assign,
                    op=op,
                    pe_idx=pe_idx,
                    start_time=start_time,
                    end_time=start_time + LATENCIES[op],
                )
            for i, dep in enumerate(deps):
                if dep not in G.nodes:
                    raise Exception

                G.add_edge(dep, assign, pos=i, op=op)

    return G


def build_design(fp):
    assert "forward_regular" in fp
    G = build_regular_code_graph(fp)
    fp_dir = os.path.split(fp)[0]
    json.dump(
        {"program_graph": nx.json_graph.node_link_data(G), "op_latencies": LATENCIES},
        open(f"{fp_dir}/design.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    args = parser.parse_args()
    build_design(args.fp)
