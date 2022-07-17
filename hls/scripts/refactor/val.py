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
    args: Tuple[str]
    res: str
    res_reg: str = None
    arity: int = 0

    def __post_init__(self):
        if self.type in {OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV, OpType.GT}:
            object.__setattr__(self, "arity", 2)
        else:
            object.__setattr__(self, "arity", 1)

    def __repr__(self):
        args = ", ".join(map(str, self.args)),
        attrs = {
            "pe": self.pe_idx,
            "opr": self.type.value,
            "op_id": self.op_id,
            # "args": str(args_attr),
        }
        if self.res_reg is not None:
            attrs["res_reg"] = self.res_reg
        attrs_str = ", ".join([f'{n} = "{v}"' for n, v in attrs.items()])
        if self.type == OpType.CST:
            return f'{self.res} = {self.type.value} {args[0]} : {State.dtype}'
        else:
            return f'{self.res} = "{self.type.value}" ({", ".join(args)}) {{  {attrs_str}  }} : ({", ".join([State.dtype] * self.arity)}) -> {State.dtype}'

def make_constant(arg):
    assert isinstance(arg, (float, bool, int)), arg
    arg = str(arg)
    cst_v = Val(id=f'cst_{arg.replace(".", "")}')
    cst_op = Op(
        OpType.CST,
        pe_idx=(-1,),
        op_id=State.curr_op_id,
        args=(arg,),
        res=str(cst_v),
    )
    State.emit(cst_op)
    # TODO
    # State.add_op_res(cst_v, cst_op)
    # State.add_edge(cst_op, "CONSTANT", cst_v)
    return cst_v


def create_new_op(op_type: OpType, args, *, pe_idx=None, res=None, add_aux_dep=False, res_reg=None):
    if pe_idx is None:
        pe_idx = State.pe_idx
    if res is None:
        res = Val()

    for i, arg in enumerate(args):
        if not isinstance(arg, Val):
            assert isinstance(arg, (float, bool, int)), arg
            args[i] = make_constant(arg)

    op = Op(op_type, pe_idx=pe_idx, op_id=State.curr_op_id, args=args, res=res, res_reg=res_reg)

    for arg in args:
        if "cst" in arg.id: continue
        State.add_edge(op, arg, op.res)

    State.emit(op)

    if add_aux_dep:
        State.maybe_add_aux_dep(pe_idx, op)

    State.maybe_add_op(op)
    State.map_val_to_pe(res, pe_idx)
    State.add_op_res(res, op)

    State.incr_op_id()

    return res


def overload_op(type):
    def f(*args: "Tuple[Val]"):
        return create_new_op(type, args)

    return f


@dataclass(frozen=True)
class Val:
    name: str = ""
    id: str = None

    def __post_init__(self):
        State.incr_var()
        if self.id is None:
            object.__setattr__(self, "id", str(State.curr_var_id))

    __add__ = overload_op(OpType.ADD)
    __sub__ = overload_op(OpType.SUB)
    __mul__ = overload_op(OpType.MUL)
    __truediv__ = overload_op(OpType.DIV)
    __gt__ = overload_op(OpType.GT)
    __neg__ = overload_op(OpType.NEG)

    def __repr__(self):
        return f"{State.val_prefix}val_{self.id}"


