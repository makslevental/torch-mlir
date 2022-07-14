from enum import Enum
from textwrap import dedent, indent
from typing import Tuple

from dataclasses import dataclass

import hls.scripts.refactor.state as state


class OpType(Enum):
    ADD = "fadd"
    SUB = "fsub"
    MUL = "fmul"
    DIV = "fdiv"
    GT = "fcmpugt"
    NEG = "fneg"
    RELU = "frelu"
    CST = "arith.constant"


LATENCIES = {
    OpType.ADD: 4,
    OpType.SUB: 4,
    OpType.MUL: 3,
    OpType.DIV: 3,
    OpType.GT: 1,
    OpType.NEG: 1,
    OpType.RELU: 1,
    OpType.CST: 0,
}


def make_latency_attrs():
    # deps = sorted([list(dep) for dep in DEPS])
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
    key: OpType
    pe_idx: Tuple[int, ...]
    arity: int = 0

    def __post_init__(self):
        if self.key in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.GT}:
            object.__setattr__(self, "arity", 2)
        else:
            object.__setattr__(self, "arity", 1)

    def __repr__(self):
        return f'"{self.key.value}" (ARGS) {{ pe = "{self.pe_idx}", opr = "{self.key.value}" }} : ({", ".join([state.DTYPE] * self.arity)}) -> {state.DTYPE}'


@dataclass(frozen=True)
class Val:
    name: str = ""
    id: str = str(state.VAR_COUNT)

    def __post_init__(self):
        state.VAR_COUNT += 1
        object.__setattr__(self, "id", str(state.VAR_COUNT))

    def __overload_op(type):
        def f(*args: "Tuple[Val]"):
            state.OP_CALL_COUNT += 1

            op = Op(type, state.PE_IDX)
            if op not in state.OP_GRAPH.nodes:
                state.OP_GRAPH.add_node(op)
            v = Val()
            state.VAL_SOURCE[v] = op

            arg_strs = []
            for arg in args:
                if not isinstance(arg, Val):
                    raise Exception(f"unknown val {arg}")
                    arg = Val(name=str(arg))
                arg_strs.append(str(arg))
                if arg not in state.VAL_SOURCE:
                    raise Exception(f"val source not found for {arg}")
                val_source = state.VAL_SOURCE[arg]
                state.OP_GRAPH.add_edge(
                    val_source, op, input=arg, output=v, id=state.OP_CALL_COUNT
                )

            op_str = str(op)
            state.emit(f"{v} = {op_str.replace('ARGS', ', '.join(arg_strs))}")

            return v

        return f

    __add__ = __overload_op(OpType.ADD)
    __sub__ = __overload_op(OpType.SUB)
    __mul__ = __overload_op(OpType.MUL)
    __truediv__ = __overload_op(OpType.DIV)
    __gt__ = __overload_op(OpType.GT)
    __neg__ = __overload_op(OpType.NEG)

    def __repr__(self):
        return f"{state.VAL_PREFIX}val_{self.id}"


def make_constant(val):
    value = Val(str(val))
    state.VAL_SOURCE[value] = state.CONSTANT
    state.emit(f"{value} = arith.constant {val} : {state.DTYPE}")
    return value
