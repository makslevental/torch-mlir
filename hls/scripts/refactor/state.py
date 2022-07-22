import logging
import os

import networkx as nx

logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
logger = logging.getLogger(__name__)

INPUT = "INPUT"
MEMREF_ARG = "MEMREF_ARG"
GLOBAL_MEMREF_ARG = "GLOBAL_MEMREF_ARG"
CONSTANT = "CONSTANT"
VAL_PREFIX = "%"
DTYPE = "f32"
COLLAPSE_MACS = int(os.environ.get("COLLAPSE_MACS", 0))
logger.debug(f"{COLLAPSE_MACS=}")
DEBUG = False
INCLUDE_AUX_DEPS = True


class State:
    _var_count = 0
    _op_call_count = 0
    op_graph = nx.MultiDiGraph()
    cst_map = {}
    cst_count = 0
    _pe_idx = (0,)
    val_source = {}
    val_to_pe_idx = {}
    pe_idx_to_most_recent_op_id = {}
    op_id_to_pe_idx = {}
    pe_deps = set()

    def __init__(self, output_fp):
        self.op_graph.add_nodes_from([INPUT, MEMREF_ARG, GLOBAL_MEMREF_ARG, CONSTANT])
        self.output_file = open(output_fp, "w")

    def set_output_file(self, fp):
        self.output_file = open(fp, "w")

    def incr_var(self):
        self._var_count += 1

    @property
    def curr_var_id(self):
        return self._var_count

    def incr_op_id(self):
        self._op_call_count += 1

    @property
    def curr_op_id(self):
        return self._op_call_count

    def emit(self, *args):
        print(*args, file=self.output_file)

    def debug_print(self, *args):
        if DEBUG:
            self.emit(["//"] + args)

    def add_val_source(self, v, src):
        self.val_source[v] = src

    def add_global_memref_arg(self, v):
        self.val_source[v] = GLOBAL_MEMREF_ARG

    def add_memref_arg(self, v):
        self.val_source[v] = MEMREF_ARG

    def add_constant(self, v):
        self.val_source[v] = CONSTANT

    def add_op_res(self, v, op):
        self.val_source[v] = op

    def maybe_add_op(self, op):
        if op not in self.op_graph.nodes:
            self.op_graph.add_node(op)

    def add_edge(self, op, arg, out_v):
        val_source = self.get_arg_src(arg)
        self.op_graph.add_edge(val_source, op, input=arg, output=out_v, id=op.op_id)

    def update_most_recent_pe_idx(self, pe_idx, op):
        self.pe_idx_to_most_recent_op_id[pe_idx] = op.op_id

    def get_most_recent_op_id(self, pe_idx):
        return self.pe_idx_to_most_recent_op_id[pe_idx]

    def maybe_add_aux_dep(self, pe_idx, op):
        if pe_idx in self.pe_idx_to_most_recent_op_id:
            prev_op_id = self.get_most_recent_op_id(pe_idx)
            self.pe_deps.add((prev_op_id, op.op_id))
        self.update_most_recent_pe_idx(pe_idx, op)

    def get_arg_src(self, arg):
        assert arg in self.val_source
        return self.val_source[arg]

    @property
    def dtype(self):
        return DTYPE

    @property
    def include_aux_deps(self):
        return INCLUDE_AUX_DEPS

    @property
    def val_prefix(self):
        return VAL_PREFIX

    @property
    def collapse_macs(self):
        return COLLAPSE_MACS

    @property
    def pe_idx(self):
        return self._pe_idx

    @pe_idx.setter
    def pe_idx(self, x):
        self._pe_idx = x

    def map_val_to_pe(self, v, pe_idx):
        self.val_to_pe_idx[v] = pe_idx

    def swap_output_file(self, new_file):
        old_file = self.output_file
        self.output_file = new_file
        return old_file

    def read_output_file(self):
        self.output_file.seek(0)
        return self.output_file.read()

    @property
    def num_unique_pes(self):
        return len(set(self.val_to_pe_idx.values()))

    def __del__(self):
        self.output_file.close()


state = None


def idx_to_str(idx):
    return "_".join(map(str, idx))
