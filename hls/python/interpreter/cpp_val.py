from __future__ import annotations

import io
import itertools
import os
from enum import Enum
from typing import Dict, List

import numpy as np

from hls.python.interpreter.util import (
    get_default_args,
    index_map,
    format_cst,
    idx_to_id,
)

VAR_COUNT = 0
OP_COUNT = 0
OP_COUNT_ZFILL = 5
FILE = None
OP_LAYER_PREFIX = "layer0"


def make_op_id():
    return f"{OP_LAYER_PREFIX}_{f'{OP_COUNT}'.zfill(OP_COUNT_ZFILL)}"


class Op(Enum):
    MUL = "mul"
    DIV = "div"
    ADD = "add"
    SUB = "sub"
    GT = "cmpugt"
    NEG = "neg"
    MULADD = "muladd"
    CST = "cst"
    RELU = "relu"


BIN_OPS = {Op.MUL, Op.DIV, Op.ADD, Op.SUB, Op.GT}


def emitter(op: Op, res_v: CPPVal, *operands):
    global OP_COUNT
    OP_COUNT += 1

    if op in BIN_OPS:
        a, b = operands
        print(
            f"float {res_v} = f{op.value}_{make_op_id()}({a}, {b});",
            file=FILE,
        )
    elif op == Op.NEG:
        (a,) = operands
        print(
            f"float {res_v} = fneg_{make_op_id()}{a};",
            file=FILE,
        )
    elif op == Op.RELU:
        (a,) = operands
        print(
            f"float {res_v} = frelu_{make_op_id()}{a};",
            file=FILE,
        )
    elif op == Op.CST:
        cst, = operands
        print(f"float {res_v} = {cst};", file=FILE)
    elif op == Op.MULADD:
        a, b, c = operands
        print(
            f"float {res_v} = fmuladd_{make_op_id()}({a}, {b}, {c});",
            file=FILE,
        )
    else:
        raise Exception("unknown op")


class CPPVal:
    def __init__(self, name):
        global VAR_COUNT
        VAR_COUNT += 1

        self.name = f"{name}"
        self.val_id = f"val_{VAR_COUNT}"

    def __mul__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(* {self} {other})")
        emitter(Op.MUL, v, self, other)
        return v

    def __truediv__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(/ {self} {other})")
        emitter(Op.DIV, v, self, other)
        return v

    def __add__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(+ {self} {other})")
        emitter(Op.ADD, v, self, other)
        return v

    def __sub__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(- {self} {other})")
        emitter(Op.SUB, v, self, other)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(> {self} {other})")
        emitter(Op.GT, v, self, other)
        return v

    def __neg__(self):
        v = CPPVal(f"(- {self})")
        emitter(Op.NEG, v, self)
        return v

    def __str__(self):
        return f"{self.val_id}"


class CPPConstant(CPPVal):
    def __init__(self, name):
        super(CPPConstant, self).__init__(name)
        self.fmted_cst = f"{format_cst(self.name)}"
        emitter(Op.CST, self, self.fmted_cst)

    def __str__(self):
        return f"cst_{self.fmted_cst}"


class ArrayVal(CPPVal):
    def __init__(self, arr_name, val_idx):
        super().__init__(arr_name)
        self.val_id = idx_to_id(val_idx)

    def __str__(self):
        return f"{self.name}_{self.val_id}"


def replace_trailing_num(global_name):
    # repeat constants with the same shape have alpha suffices
    reps = ["a", "b", "c", "d", "e", "f"]
    for i, rep in enumerate(reps):
        if f"f32_{i}" in global_name:
            return global_name.replace(f"f32_{i}", f"f32_{rep}")
    return global_name


class Array:
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

    def __getitem__(self, index):
        index = self.idx_map(index)
        if index not in self.registers:
            self.registers[index] = self.make_val(index)
        v = self.registers[index]
        return v

    def make_val(self, index):
        name = self.arr_name
        if self.output:
            name = f"output_{name}"
        v = ArrayVal(name, index)
        return v

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class GlobalArray(Array):
    def __init__(self, _arr_name, global_name, global_array):
        super().__init__(global_name, *global_array.shape)
        self.registers = {}
        self._csts = {}
        for idx in np.ndindex(global_array.shape):
            self.registers[idx] = ArrayVal(self.arr_name, idx)
            self._csts[idx] = global_array[idx]

    def __getitem__(self, index):
        return self.registers[index]


def FMulAdd(a, b, c):
    inps = [a, b, c]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    a, b, c = inps
    v = CPPVal(f"(fmuladd {a} {b} {c})")
    emitter(Op.MULADD, v, a, b, c)
    return v


def FMac(a, b, arr, idx):
    inps = [a, b]
    for i, v in enumerate(inps):
        if isinstance(v, (float, int, bool)):
            inps[i] = CPPConstant(v)
    (a, b), c = inps, arr[idx]
    cc = c

    # handle the first time you process a constant of input
    if isinstance(c, ArrayVal):
        cc = arr[idx] = arr.make_val(idx)

    emitter(Op.MULADD, cc, a, b, c)
    return cc


def Add(a, arr, idx):
    if isinstance(a, (float, int, bool)):
        a = CPPConstant(a)

    # handle the first time you process a constant of input
    b = bb = arr[idx]
    if isinstance(b, (ArrayVal, CPPConstant)):
        bb = arr[idx] = arr.make_val(idx)

    emitter(Op.ADD, bb, a, b)
    return bb


def ReduceAdd(src_arr: Array, dst_arr: Array):
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


def Copy(a: Array, b: Array):
    assert isinstance(b, Array)
    a.registers = b.registers


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


def ReLU(x):
    v = CPPVal(f"(relu {x})")
    emitter(Op.RELU, v, x)
    return v


def split_args(args):
    inps, outps, globs = [], [], []
    for _arg_name, arg in args.items():
        if isinstance(arg, GlobalArray):
            globs.append(arg)
        elif isinstance(arg, Array):
            if arg.input:
                inps.append(arg)
            elif arg.output:
                outps.append(arg)
        else:
            raise Exception("unknown forward arg")

    return inps, outps, globs


def splat_args(inputs: List[Array], outputs: List[Array], globals: List[GlobalArray]):
    args = []
    for inp in inputs + globals:
        for idx in itertools.product(*[range(s) for s in inp.curr_shape]):
            id = idx_to_id(idx)
            args.append(f"float {inp.arr_name}_{id}")
    for outp in outputs:
        for idx in itertools.product(*[range(s) for s in outp.curr_shape]):
            id = idx_to_id(idx)
            args.append(f"float* {outp.arr_name}_{id}")

    return args


def CPPForward(args: Dict, forward):
    global FILE
    fn = os.environ.get("FN", "regular")
    FILE = open(f"forward{f'_{fn}' if fn else ''}.cpp", "w")

    inputs, outputs, globals = split_args(args)

    OLD_FILE = FILE
    FILE = io.StringIO()
    print(
        'extern "C" '
        + f"void forward({', '.join(splat_args(inputs, outputs, globals))}) {{\n",
        file=FILE,
    )

    forward()

    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    assert len(outputs) == 1, f"wtfbbq {len(outputs)}"
    output = outputs[0]
    for index, value in output.registers.items():
        id = "_".join(map(str, index))
        print(f"*{output.arr_name}_{id} = {value};", file=OLD_FILE)

    print("return;", file=OLD_FILE)
    print("}", file=OLD_FILE)

    OLD_FILE.close()


def Forward(forward):
    args = get_default_args(forward)
    CPPForward(args, forward)
