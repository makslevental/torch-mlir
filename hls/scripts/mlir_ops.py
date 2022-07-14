from __future__ import annotations

import _ast
import argparse
import ast
import glob
import inspect
from _ast import Subscript
from ast import Assign, Mult, Add, BinOp, Name, Call, keyword, Str, IfExp, Compare
from typing import Tuple, Union, Dict, Any

import astor
import numpy as np

MAC_IDX = None

OUTPUT_ARRAYS = []

from dataclasses import dataclass, field

ArrayIndex = Tuple[int]

PEIndex = Tuple[int]

DTYPE = "half"


@dataclass(frozen=True)
class _Instruction:
    pe_id: PEIndex = field(init=False, default_factory=lambda: CURRENT_PE)


@dataclass(frozen=True)
class NOP(_Instruction):
    pass


@dataclass(frozen=True)
class Bin(_Instruction):
    left: Any
    right: Any


@dataclass(frozen=True)
class MulInst(Bin):
    pass


@dataclass(frozen=True)
class DivInst(Bin):
    pass


@dataclass(frozen=True)
class AddInst(Bin):
    pass


@dataclass(frozen=True)
class SubInst(Bin):
    pass


@dataclass(frozen=True)
class GTInst(Bin):
    pass


@dataclass(frozen=True)
class ReLUInst(_Instruction):
    val: Any


@dataclass(frozen=True)
class ExpInst(_Instruction):
    val: Any


Instruction = Union[NOP, ExpInst, ReLUInst, MulInst, AddInst, DivInst, SubInst, GTInst]

# def assert_never(x: NoReturn) -> NoReturn:
#     raise AssertionError("Unhandled type: {}".format(type(x).__name__))
#
# def showResult(r: Result) -> str:
#     if isinstance(r, OK):
#         return str(r.result)
#     elif isinstance(r, Failure):
#         return "Failure: " + r.msg
#     else:
#         assert_never(r)

CURRENT_PE: PEIndex = None


class PE:
    def __init__(self, index):
        self.index = index
        self._instructions = []

    def push_instructions(self, inst):
        self._instructions.append(inst)

    def num_instructions(self):
        return len(self._instructions)

    def push_nop(self):
        self._instructions.append(NOP())


PES: Dict[PEIndex, PE] = {}


def index_map(index, curr_shape, prev_shape):
    return tuple(np.unravel_index(np.ravel_multi_index(index, curr_shape), prev_shape))


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_array_type(shape, ptr=True, nd=False):
    if nd:
        typ = ""
        for s in shape:
            typ += f"[{s} x "
        typ += DTYPE
        for _ in shape:
            typ += "]"
        if ptr:
            typ += "*"
    else:
        typ = f"{np.prod(shape)} x {DTYPE}"
    return typ


