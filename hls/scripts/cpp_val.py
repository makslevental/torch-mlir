import argparse
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


class CPPVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.val_id = f"val_{VAR_COUNT}"
        VAR_COUNT += 1

    def __mul__(self, other):
        global OP_COUNT
        OP_COUNT += 1
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(* ({self}) ({other}))")
        print(
            f"float {v} = fmul_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self}, {other});",
            file=FILE,
        )

        return v

    def __add__(self, other):
        global OP_COUNT
        OP_COUNT += 1
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(+ ({self}) ({other}))")
        print(
            f"float {v} = fadd_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self}, {other});",
            file=FILE,
        )
        return v

    def __sub__(self, other):
        global OP_COUNT
        OP_COUNT += 1
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(- ({self}) ({other}))")
        print(
            f"float {v} = fsub_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self}, {other});",
            file=FILE,
        )
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        global OP_COUNT
        OP_COUNT += 1
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(/ ({self}) ({other}))")
        print(
            f"float {v} = fdiv_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self}, {other});",
            file=FILE,
        )
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        global OP_COUNT
        OP_COUNT += 1
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(> ({self}) ({other}))")
        print(
            f"float {v} = fcmpugt_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self}, {other});",
            file=FILE,
        )
        return v

    def __neg__(self):
        # <result> = fcmp ugt float 4.0, 5.0
        global OP_COUNT
        OP_COUNT += 1
        v = CPPVal(f"(- {self})")
        print(
            f"float {v} = fneg_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({self});",
            file=FILE,
        )
        return v

    def __str__(self):
        return f"{self.val_id}"


class CPPConstant(CPPVal):
    def __init__(self, name):
        super(CPPConstant, self).__init__(name)
        self.fmted_cst = f"{format_cst(self.name)}"
        print(f"float {self} = {name};", file=FILE)

    def __str__(self):
        return f"cst{self.fmted_cst}"


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
            value = CPPConstant(value)
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


class ArrayVal(CPPVal):
    array: ArrayDecl

    def __init__(self, arr_name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(arr_name)
        self.array = array
        self.val_id = "_".join(map(str, val_id))

    def __str__(self):
        return f"{self.name}_{self.val_id}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"{self.name}_{self.val_id}"


def FMulAdd(a, b, c):
    global OP_COUNT
    OP_COUNT += 1

    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    a, b, c = inps
    v = CPPVal(f"(fmuladd {a} {b} {c})")
    print(
        f"float {v} = fmuladd_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({a}, {b}, {c});",
        file=FILE,
    )
    return v


def FMac(a, b, arr, idx):
    global OP_COUNT
    OP_COUNT += 1

    inps = [a, b]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    (a, b), c = inps, arr[idx]
    cc = c

    # handle the first time you process a constant of input
    if isinstance(c, GlobalArrayVal):
        cc = arr[idx] = arr.make_val(idx)
    print(
        f"float {cc} = fmuladd_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({a}, {b}, {c});",
        file=FILE,
    )
    return cc


def Add(a, arr, idx):
    global OP_COUNT
    OP_COUNT += 1

    if isinstance(a, (float, int, bool)):
        a = CPPConstant(a)

    # handle the first time you process a constant of input
    b = bb = arr[idx]
    if isinstance(b, (GlobalArrayVal, CPPConstant)):
        bb = arr[idx] = arr.make_val(idx)
    print(
        f"float {bb} = fadd_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({a}, {b});", file=FILE
    )
    return bb


def ReduceAdd(src_arr: ArrayDecl, dst_arr: ArrayDecl):
    prev_sums = list(src_arr.registers.values())
    while len(prev_sums) > 1:
        next_sums = []
        while len(prev_sums) > 1:
            next_sums.append(prev_sums.pop() + prev_sums.pop())
        if len(prev_sums) == 1:
            next_sums[-1] = next_sums[-1] + prev_sums[0]
        prev_sums = next_sums
    dst_arr[
        0,
    ] = prev_sums[0]


def Copy(a: ArrayDecl, b: ArrayDecl):
    assert isinstance(b, ArrayDecl)
    a.registers = b.registers


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


def ReLU(x):
    global OP_COUNT
    OP_COUNT += 1
    v = CPPVal(f"(relu {x})")
    print(f"float {v} = relu_{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}({x});", file=FILE)
    return v


def make_args_globals(Args):
    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                if arg.input:
                    args.append(f"float {arg.arr_name}_{id}")
                elif arg.output:
                    args.append(f"float* {arg.arr_name}_{id}")
        elif isinstance(arg, GlobalArray):
            for idx in itertools.product(*[range(s) for s in arg.curr_shape]):
                id = "_".join(map(str, idx))
                globals.append(f"float {arg.arr_name}_{id}")

    return args, globals


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
    fn = os.environ.get("FN", "regular")
    FILE = open(f"forward{f'_{fn}' if fn else ''}.cpp", "w")

    args, globals = make_args_globals(Args)

    OLD_FILE = FILE
    FILE = io.StringIO()
    print('extern "C" ' + f"void forward({', '.join(args + globals)}) {{\n", file=FILE)

    forward()

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for index, value in output.registers.items():
        id = "_".join(map(str, index))
        print(f"*{output.arr_name}_{id} = {value};", file=OLD_FILE)

    print("return;", file=OLD_FILE)
    print("}", file=OLD_FILE)

    OLD_FILE.close()


def Forward(forward, max_range=None, worker_id=None):
    Args = get_default_args(forward)
    CPPForward(Args, Args["_arg1"], forward)


def get_ssas_from_ir_line(line):
    line = re.sub(r", align \d+", "", line)

    idents = list(
        filter(
            lambda x: len(x.strip())
            and "mul" not in x
            and "add" not in x
            and "relu" not in x
            and "neg" not in x,
            [
                f
                for f, _ in re.findall(r"([\d|a-z|_]*|([0-9]*[.])+[0-9]+)", line)
                if len(f.strip())
                and f.strip() not in {"void", "forward", "float", "extern", "return"}
            ],
        )
    )

    if not idents or "declare" in line or "//" in line:
        return None, None, None

    if (
        "fmul" in line
        or "fadd" in line
        or "fneg" in line
        or "fcmp" in line
        or "fsub" in line
        or "fdiv" in line
        or "relu" in line
        or "fmuladd" in line
    ):
        assign, *_deps = idents
        deps = []
        for d in _deps:
            try:
                float(d)
            except:
                deps.append(d)
        op = line.split("=")[1].strip().split("(")[0]
    elif "*" in line and "=" in line:
        assign, dep = idents
        deps = [dep]
        op = "store"
    elif "expf" in line:
        assign, *deps = idents
        op = "expf"
    elif "forward" in line:
        inputs = [
            f.replace("float ", "")
            for f, _ in re.findall(r"(float ([\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace("float* ", "")
            for f, _ in re.findall(r"(float\* ([\d|a-z|_]*))", line)
        ]
        return inputs, outputs, ""
    elif "float" in line and "cst" in line:
        return None, None, None
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


def build_regular_code_graph(fp):
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
                        assert (
                            "__constant" in dep or "input" in dep or "cst" in dep
                        ), dep
                        if "input" in dep:
                            G.add_node(dep, op="input")
                        elif "cst" in dep:
                            G.add_node(dep, op="constant")
                        elif "__constant" in dep:
                            G.add_node(dep, op="constant")

                    G.add_edge(dep, assign, pos=i, op=op)

    return G


def build_macs_graph(fp):
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
                first_assign = False
                if assign not in G.nodes:
                    G.add_node(assign, op=op)
                    first_assign = True
                for i, dep in enumerate(deps):
                    if dep not in G.nodes or (dep == assign and first_assign):
                        assert (
                            "__constant" in dep or "input" in dep or "cst" in dep
                        ), dep
                        if "input" in dep:
                            G.add_node(dep, op="input")
                        elif "cst" in dep:
                            G.add_node(dep, op="constant")
                        elif "__constant" in dep:
                            glob = re.search(r"__constant_(.*f32)((_\d+)+)", dep)
                            assert glob
                            glob_dep = f"glob_{glob[0]}"
                            assert G.nodes[glob_dep]["op"] == "input"
                            dep = glob_dep

                    G.add_edge(dep, assign, pos=i, op=op)

    return G


def build_op_topo_sort(G):
    topo_sort = []
    for i, stage in enumerate(topological_sort_grouped(G)):
        ops = []
        for val in stage:
            op = G.nodes[val]["op"]
            if op not in {"input", "output", "constant"}:
                ops.append(op)
        if ops:
            topo_sort.append(sorted(ops))

    return topo_sort


def build_design(fp):
    assert "forward_regular" in fp
    G = build_regular_code_graph(fp)
    op_topo = build_op_topo_sort(G)
    print("num stages", len(op_topo))
    print(list(map(lambda x: len(x), op_topo)))

    G = build_macs_graph(fp.replace("regular", "macs"))

    design = {
        "G": nx.json_graph.node_link_data(G),
        "topo_sort": op_topo,
    }

    fp_dir = os.path.split(fp)[0]
    json.dump(design, open(f"{fp_dir}/design.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    args = parser.parse_args()
    build_design(args.fp)
