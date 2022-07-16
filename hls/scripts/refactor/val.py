from enum import Enum
from textwrap import dedent, indent
from typing import Tuple

from dataclasses import dataclass

from hls.scripts.refactor.state import state as State


class OpType(Enum):
    ADD = "fadd"
    SUB = "fsub"
    MUL = "fmul"
    DIV = "fdiv"
    GT = "fcmpugt"
    NEG = "fneg"
    RELU = "frelu"
    CST = "arith.constant"
    COPY = "copy"


LATENCIES = {
    OpType.ADD: 4,
    OpType.SUB: 4,
    OpType.MUL: 3,
    OpType.DIV: 3,
    OpType.GT: 1,
    OpType.NEG: 1,
    OpType.RELU: 1,
    OpType.CST: 0,
    OpType.COPY: 1,
}


def make_latency_attrs():
    if State.include_aux_deps:
        aux_deps = sorted([list(dep) for dep in State.pe_deps])
    else:
        aux_deps = []
    operator_types = [
        f"""{{ name = "{op.value}", latency = {lat}, limit = 2592 }}"""
        for op, lat in LATENCIES.items()
    ]
    lats = indent(
        dedent(
            f"""
         attributes {{
          auxdeps = {aux_deps},
          operatortypes = [
            {', '.join(operator_types)}
            ] }} \n{{
        """,
        ),
        "\t",
    )
    return lats


@dataclass(frozen=True)
class Op:
    type: OpType
    pe_idx: Tuple[int, ...]
    op_id: int
    arity: int = 0

    def __post_init__(self):
        if self.type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.GT}:
            object.__setattr__(self, "arity", 2)
        else:
            object.__setattr__(self, "arity", 1)

    def __repr__(self):
        return f'"{self.type.value}" (ARGS) {{ pe = "{self.pe_idx}", opr = "{self.type.value}", op_id = "{self.op_id}" }} : ({", ".join([State.dtype] * self.arity)}) -> {State.dtype}'


def create_new_op(op_type: OpType, pe_idx=None, res=None, add_aux_dep=False):
    if pe_idx is None:
        pe_idx = State.pe_idx
    if res is None:
        res = Val()

    op = Op(op_type, pe_idx, State.curr_op_id)

    if add_aux_dep:
        State.maybe_add_aux_dep(pe_idx, op)

    State.maybe_add_op(op)
    State.map_val_to_pe(res, pe_idx)
    State.add_op_res(res, op)

    State.incr_op_id()

    return op, res


def create_new_op_arg(op, arg, op_res):
    if not isinstance(arg, Val):
        assert isinstance(arg, (float, bool, int)), arg
        arg = Constant(arg)
    State.add_edge(op, arg, op_res)
    return arg


def overload_op(type):
    def f(*args: "Tuple[Val]"):
        op, op_res = create_new_op(type)
        arg_strs = []
        for arg in args:
            arg = create_new_op_arg(op, arg, op_res)
            arg_strs.append(str(arg))

        op_str = str(op)
        State.emit(f"{op_res} = {op_str.replace('ARGS', ', '.join(arg_strs))}")

        return op_res

    return f


@dataclass(frozen=True)
class Val:
    name: str = ""
    id: str = str(State.curr_var_id)

    def __post_init__(self):
        State.incr_var()
        object.__setattr__(self, "id", str(State.curr_var_id))

    __add__ = overload_op(OpType.ADD)
    __sub__ = overload_op(OpType.SUB)
    __mul__ = overload_op(OpType.MUL)
    __truediv__ = overload_op(OpType.DIV)
    __gt__ = overload_op(OpType.GT)
    __neg__ = overload_op(OpType.NEG)

    def __repr__(self):
        return f"{State.val_prefix}val_{self.id}"


def Constant(cst):
    assert isinstance(cst, (float, bool, int)), cst
    op, op_res = create_new_op(OpType.CST, pe_idx=(-1,))
    op_str = str(op)
    State.emit(f"{op_res} = {op_str.replace('ARGS', str(cst))}")

    return op_res


ReLU = lambda x: overload_op(OpType.RELU)(x)
Copy = lambda x: overload_op(OpType.COPY)(x)
