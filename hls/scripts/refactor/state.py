import os

import networkx as nx

UNIQUE_IDXS = set()
VAR_COUNT = 0
OP_CALL_COUNT = 0
VAL_PREFIX = "%"
OP_GRAPH = nx.MultiDiGraph()

INPUT = "INPUT"
MEMREF_ARG = "MEMREF_ARG"
GLOBAL_MEMREF_ARG = "GLOBAL_MEMREF_ARG"

CONSTANT = "CONSTANT"
CST_MAP = {}
CST_COUNT = 0
OP_GRAPH.add_node(INPUT)
PE_IDX = (0,)
DTYPE = "f32"

COLLAPSE_MACS = False

VAL_SOURCE = {}
VAL_TO_PE_IDX = {}

FN = os.environ.get("FN", "regular")
OUTPUT_FILE = open(f"forward{f'_{FN}' if FN else ''}.mlir", "w")


def emit(*args):
    print(*args, file=OUTPUT_FILE)
