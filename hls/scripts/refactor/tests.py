from pprint import pprint
import networkx as nx
import numpy as np

from hls.scripts.refactor.memref import MemRefVal, MemRef, GlobalMemRef
from hls.scripts.refactor.val import Val
import hls.scripts.refactor.state as state


def test1():
    n = Val(name="inp_a")
    state.VAL_SOURCE[n] = state.INPUT
    m = Val(name="inp_b")
    state.VAL_SOURCE[m] = state.INPUT
    p = Val(name="inp_c")
    state.VAL_SOURCE[p] = state.INPUT

    o = n * m
    print("o", o)
    q = o * p
    print("q", q)
    r = o + q
    print("r", r)

    pprint(nx.to_dict_of_dicts(state.OP_GRAPH))

def test2():
    a = MemRefVal(name="arr1", id=(0, 1))
    state.VAL_SOURCE[a] = state.INPUT
    b = MemRefVal("arr2", (0, 2))
    state.VAL_SOURCE[b] = state.INPUT

    _arg0 = MemRef("_arg0", 3, 3, input=True)
    print(_arg0.val_names)
    _arg1 = MemRef("_arg1", 1, output=True)
    print(_arg1.val_names)

    __constant_1x3xf32 = np.array([[1, 2, 3]])
    __constant_3xf32 = np.array([-1, -2, -3])

    _0 = GlobalMemRef("__constant_1x3xf32_1", __constant_1x3xf32)
    print(_0.val_names)
    _1 = GlobalMemRef("__constant_3xf32_2", __constant_3xf32)
    print(_1.val_names)

    x = _0[0, 0] * _1[0]
    y = _0[0, 0] * _1[1]
    state.PE_IDX = (1, 1)
    z = x + y

    # pprint(nx.to_dict_of_dicts(OP_GRAPH))



if __name__ == "__main__":
    test1()
    test2()
