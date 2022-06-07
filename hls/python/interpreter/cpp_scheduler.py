import argparse
import json
import os
import re

import networkx as nx

from hls.python.interpreter.util import get_ssas_from_ir_line, topological_sort_grouped


def build_regular_code_graph(fp):
    lines = open(fp, "r").readlines()
    G = nx.MultiDiGraph()

    for line in lines:
        assign, deps, op = get_ssas_from_ir_line(line)
        if "forward" in line:
            for assig in assign:
                G.add_node(assig, op="input")
            for dep in deps:
                G.add_node(dep, op="output")
        else:
            if assign is not None:
                if assign not in G.nodes:
                    G.add_node(assign, op=op)
                for i, dep in enumerate(deps):
                    if dep not in G.nodes:
                        assert (
                                "__constant" in dep or "input" in dep or "cst" in dep
                        ), dep
                        if "input" in dep:
                            G.add_node(dep, op="input")
                        elif "cst" in dep:
                            G.add_node(dep, op="constant")
                        elif "__constant" in dep:
                            G.add_node(dep, op="constant")

                    G.add_edge(dep, assign, pos=i, op=op)

    return G


def build_macs_graph(fp):
    lines = open(fp, "r").readlines()
    G = nx.MultiDiGraph()

    for line in lines:
        assign, deps, op = get_ssas_from_ir_line(line)
        if "forward" in line:
            for assig in assign:
                G.add_node(assig, op="input")
            for dep in deps:
                G.add_node(dep, op="output")
        else:
            if assign is not None:
                first_assign = False
                if assign not in G.nodes:
                    G.add_node(assign, op=op)
                    first_assign = True
                for i, dep in enumerate(deps):
                    if dep not in G.nodes or (dep == assign and first_assign):
                        assert (
                                "__constant" in dep or "input" in dep or "cst" in dep
                        ), dep
                        if "input" in dep:
                            G.add_node(dep, op="input")
                        elif "cst" in dep:
                            G.add_node(dep, op="constant")
                        elif "__constant" in dep:
                            glob = re.search(r"__constant_(.*f32)((_\d+)+)", dep)
                            assert glob
                            glob_dep = f"glob_{glob[0]}"
                            assert G.nodes[glob_dep]["op"] == "input"
                            dep = glob_dep

                    G.add_edge(dep, assign, pos=i, op=op)

    return G


def build_op_topo_sort(G):
    topo_sort = []
    for i, stage in enumerate(topological_sort_grouped(G)):
        ops = []
        for val in stage:
            op = G.nodes[val]["op"]
            if op not in {"input", "output", "constant"}:
                ops.append(op)
        if ops:
            topo_sort.append(sorted(ops))

    return topo_sort


def build_design(fp):
    assert "forward_regular" in fp
    G = build_regular_code_graph(fp)
    op_topo = build_op_topo_sort(G)

    G = build_macs_graph(fp.replace("regular", "macs"))

    design = {
        "G": nx.json_graph.node_link_data(G),
        "topo_sort": op_topo,
    }

    fp_dir = os.path.split(fp)[0]
    json.dump(design, open(f"{fp_dir}/design.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    args = parser.parse_args()
    build_design(args.fp)
