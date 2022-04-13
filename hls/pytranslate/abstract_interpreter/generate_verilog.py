import argparse
import re
from collections import defaultdict
from pathlib import Path

from toposort import toposort

mul_adder = (
    lambda id, din0, din1, din2, dout: f"""
forward_fmul_32ns_32ns_32_10_med_dsp_1 #(
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
fmul_32ns_32ns_32_10_med_dsp_1_U{id}(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0({din0}),
    .din1({din1}),
    .din2({din2}),
    .ce(1'b1),
    .dout({dout})
);
"""
)


def make_schedule(fp):
    lines = open(fp).readlines()

    G = defaultdict(set)
    defining_lines = {}

    for i, line in enumerate(lines):
        if "define" in line:
            continue
        idents = re.findall(r"(%[\d|a-z|_]*)", line)
        if not idents:
            continue

        if "fmul" in line or "fadd" in line:
            assign, *deps = idents
        elif "store" in line:
            dep, assign = idents
            deps = [dep]
        else:
            raise Exception("wtfbbq")

        defining_lines[assign] = line

        for dep in deps:
            G[assign].add(dep)

    stages = defaultdict(list)
    max_muls = 0
    for i, stage in enumerate(toposort(G)):
        print(f"\nstage {i}; " + "num ops " + str(len(stage)) + " " + "-" * 10 + "\n")
        muls = 0
        if len(stage):
            for var in stage:
                if var in defining_lines:
                    defining_line = defining_lines[var]
                    stages[i].append(defining_line)
                    if "mul" in defining_line:
                        muls += 1
                    max_muls = max(max_muls, muls)
    print(max_muls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make stuff")
    parser.add_argument("fp", type=Path)
    args = parser.parse_args()
    make_schedule(str(args.fp))
