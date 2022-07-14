import itertools
from typing import Tuple

import numpy as np
from dataclasses import dataclass

import hls.scripts.refactor.state as state
from hls.scripts.refactor.state import VAL_SOURCE
from hls.scripts.refactor.val import Val, make_constant


def index_map(index, curr_shape, prev_shape):
    return tuple(np.unravel_index(np.ravel_multi_index(index, curr_shape), prev_shape))


MemRefIndex = Tuple[int, ...]


def idx_to_str(idx):
    return "_".join(map(str, idx))


@dataclass(frozen=True)
class MemRefVal(Val):
    id: MemRefIndex = None

    def __post_init__(self):
        assert self.id
        object.__setattr__(self, "id", idx_to_str(self.id))

    def __repr__(self):
        return f"%{self.name}_{self.id}"


class MemRef:
    def __init__(self, name, *shape, input=False, output=False):
        self.arr_name = name
        self.curr_shape = shape
        self.prev_shape = shape
        self.pe_index = shape
        self.registers = {}
        self.input = input
        self.output = output

    def __setitem__(self, index, value):
        if not isinstance(value, Val):
            assert isinstance(value, (float, bool, int))
            value = make_constant(value)
        index = self.idx_map(index)
        assert not self.input
        self.registers[index] = value

    def __getitem__(self, index: MemRefIndex):
        index = self.idx_map(index)
        if index not in self.registers:
            self.registers[index] = self.make_val(index)
        v = self.registers[index]
        return v

    def make_val(self, index):
        name = self.arr_name
        if self.output:
            name = f"output_{name}"
        v = MemRefVal(name, index)
        VAL_SOURCE[v] = state.MEMREF_ARG
        return v

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self

    @property
    def val_names(self):
        assert self.input or self.output
        val_names = []
        if self.input:
            for idx in itertools.product(*[range(s) for s in self.curr_shape]):
                val_names.append(f"%{self.arr_name}_{idx_to_str(idx)}")
        elif self.output:
            assert len(self.registers)
            val_names = sorted([str(v) for v in self.registers.values()])

        return sorted(val_names)

    @property
    def numel(self):
        return np.prod(self.curr_shape)


class GlobalMemRef:
    def __init__(self, global_name, global_array: np.ndarray):
        self.name = global_name
        self.global_array = global_array
        self.curr_shape = global_array.shape
        self.vals = {}
        for idx, v in np.ndenumerate(global_array):
            v = MemRefVal(f"{self.name}__", idx)
            state.VAL_SOURCE[v] = state.GLOBAL_MEMREF_ARG
            self.vals[idx] = v

    @property
    def val_names(self):
        return sorted([str(v) for v in self.vals.values()])

    def __getitem__(self, index: MemRefIndex):
        if isinstance(index, int):
            index = (index,)
        return self.vals[index]

    @property
    def numel(self):
        return np.prod(self.curr_shape)
