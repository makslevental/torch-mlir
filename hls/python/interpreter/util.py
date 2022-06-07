from __future__ import annotations

import inspect
import re
import struct
from typing import Tuple

import numpy as np
from llvmlite.ir import Constant, FloatType

DTYPE = "half"


def get_default_args(func):
    # get the params for Forward that have default args, which are wrappers around the arrays
    # objects that do the memory/register tracking
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def idx_to_id(idx: Tuple[int]):
    return "_".join(map(str, idx))


def id_to_idx(id: str):
    return tuple(map(int, id.split("_")))


def index_map(index, curr_shape, prev_shape):
    return tuple(np.unravel_index(np.ravel_multi_index(index, curr_shape), prev_shape))


def parse_range_call(rang):
    lo, hi, step = [r.value for r in rang.args]
    return hi


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


# def unroll_loops():
#     passer = apply_passes([loop_unroll()], env=SymbolTable(locals(), globals()))
#     new_body = passer(body)
#     fun_tree = passer.parse(new_body)
#
#     _locals = dict(
#         zip(
#             ("_arg2", "_arg3", "_arg4", "_arg5"),
#             [cst.Integer(value=repr(i)) for i in (0, 0, 0, 0)],
#         )
#     )
#     sym_table = SymbolTable(_locals, {})
#     new_new_body = replace_symbols(fun_tree.body, sym_table)
#     open("new_new_body.py", "w").write(to_source(new_new_body))
#     fun_tree = fun_tree.with_deep_changes(fun_tree, body=new_new_body)
#     open("unrolled.py", "w").write(to_source(fun_tree))

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
