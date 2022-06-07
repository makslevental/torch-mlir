import argparse
import logging
import re
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pathlib import Path
from textwrap import dedent, indent
from threading import Barrier

NUM_THREADS = 8

FILE = open("forward.1.v", "w")

mul_adder = (
    lambda id: f"""
reg   [31:0]   dsp_{id}_din0;
reg   [31:0]   dsp_{id}_din1;
reg   [31:0]   dsp_{id}_din2;
wire  [31:0]   dsp_{id}_dout;

forward_fmuladd_32ns_32ns_32_10_med_dsp_1 #(
    .din0_WIDTH( 32 ),
    .din1_WIDTH( 32 ),
    .din2_WIDTH( 32 ),
    .dout_WIDTH( 32 ))
fmuladd_32ns_32ns_32_10_med_dsp_{id}(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(dsp_{id}_din0),
    .din1(dsp_{id}_din1),
    .din2(dsp_{id}_din2),
    .ce(1'b1),
    .dout(dsp_{id}_dout)
);

"""
)

mac = (
    lambda id: f"""


"""
)

MAX_DSPS = 9024
# MAX_DSPS = 1200
MAX_BRAMS = 4032
# MAX_BRAMS = 300
DSP_LATENCY = 8
NUM_STAGES = DSP_LATENCY + 1 + 1  # read, run, write

from dataclasses import dataclass, field
from enum import Enum


class DSPInputType(Enum):
    din0 = "din0"
    din1 = "din1"
    din2 = "din2"


@dataclass(frozen=True)
class DSPInputRegister:
    dsp_id: int = field(compare=True)
    type: DSPInputType

    def __str__(self):
        return f"dsp_{self.dsp_id}_{self.type.name}"


@dataclass(frozen=True)
class DSPOutputWire:
    dsp_id: int = field(compare=True)

    def __str__(self):
        return f"dsp_{self.dsp_id}_dout"


@dataclass(frozen=True)
class FMulAddDSP:
    id: int = field(compare=True)
    din0: DSPInputRegister = field(init=False)
    din1: DSPInputRegister = field(init=False)
    din2: DSPInputRegister = field(init=False)
    dout: DSPOutputWire = field(init=False)

    def __post_init__(self):
        din0 = DSPInputRegister(self.id, DSPInputType.din0)
        din1 = DSPInputRegister(self.id, DSPInputType.din1)
        din2 = DSPInputRegister(self.id, DSPInputType.din2)
        dout = DSPOutputWire(self.id)
        object.__setattr__(self, "din0", din0)
        object.__setattr__(self, "din1", din1)
        object.__setattr__(self, "din2", din2)
        object.__setattr__(self, "dout", dout)


def get_ssas_from_ir_line(line):
    line = re.sub(r", align \d+", "", line)
    idents = [f for f, _ in re.findall(r"(%[\d|a-z|_]*|([0-9]*[.])+[0-9]+)", line)]
    if not idents:
        logging.debug(f"line has no idents: {line}")
        return None, None

    if (
        "fmul" in line
        or "fadd" in line
        or "fcmp" in line
        or "fsub" in line
        or "fdiv" in line
    ):
        assign, *deps = idents
    elif "store" in line:
        dep, assign = idents
        deps = [dep]
    elif "expf" in line:
        assign, deps = idents
    elif "define void @forward" in line:
        inputs = [
            f.replace("float ", "")
            for f, _ in re.findall(r"(float (%[\d|a-z|_]*))", line)
        ]
        outputs = [
            f.replace("float* ", "")
            for f, _ in re.findall(r"(float\* (%[\d|a-z|_]*))", line)
        ]
        return inputs, outputs
    elif "store" in line:
        logging.error(f"unrecognized inst: {line}")

    return assign, deps


def parallel_make_graph(lines):
    G = defaultdict(set)
    Ginv = defaultdict(set)
    defining_lines = {}
    inputs = []
    outputs = []
    all_vals = set()

    def process_line(line):
        nonlocal inputs, outputs
        assign, deps = get_ssas_from_ir_line(line)
        if assign is None:
            return

        if "define" in line:
            inputs.extend(assign)
            outputs.extend(deps)
            return
        if "declare" in line:
            return

        defining_lines[assign] = line

        all_vals.add(assign)
        for dep in deps:
            all_vals.add(dep)
            G[dep].add(assign)
            Ginv[assign].add(dep)

    with ThreadPool(processes=NUM_THREADS) as pool:
        pool.map(process_line, lines)

    for inp in inputs:
        Ginv[inp]
    for outp in outputs:
        G[outp]

    all_vals = sorted(all_vals)
    return all_vals, G, Ginv, defining_lines, inputs, outputs


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def parallel_topo_sort(all, G, Ginv):
    in_degree = {}
    for a in all:
        if a in Ginv:
            in_degree[a] = len(Ginv[a])
        else:
            in_degree[a] = 0

    partitions = {i: set(split) for i, split in enumerate(split(all, NUM_THREADS))}

    depth = 0

    depths = set()

    def any_live_partitions():
        return any([len(part) > 0 for part in partitions.values()])

    window_size = 5
    prev_partition_sizes = [0] * window_size

    def sync_point():
        nonlocal depth
        prev_partition_sizes[depth % window_size] = tuple(
            [len(part) for part in partitions.values()]
        )
        if len(set(prev_partition_sizes)) == 1:
            raise Exception("wtfbbq")
        depth += 1

    barrier = Barrier(NUM_THREADS, action=sync_point)

    all_outputs = defaultdict(dict)

    def task(thread_idx):
        while any_live_partitions():
            barrier.wait()
            output = set()
            for u in partitions[thread_idx]:
                if in_degree[u] <= 0:
                    output.add(u)

            barrier.wait()
            partitions[thread_idx] -= output

            barrier.wait()
            for outp in output:
                for assign in G[outp]:
                    in_degree[assign] -= 1

            all_outputs[thread_idx][depth] = output
            depths.add(depth)

    with ThreadPool(processes=NUM_THREADS) as pool:
        ress = []
        for i in range(NUM_THREADS):
            all_outputs[i] = {}
            ress.append(pool.apply_async(task, (i,)))

        [r.wait() for r in ress]

    stages = []
    for d in sorted(depths):
        stage = set()
        for i in range(NUM_THREADS):
            stage |= all_outputs[i][d]
        if stage:
            stages.append(stage)

    return stages


def make_schedule(topo_sort_stages, defining_lines, inputs, outputs):
    dsps = {}

    def get_dsp(i) -> FMulAddDSP:
        if i not in dsps:
            dsps[i] = FMulAddDSP(i)
        return dsps[i]

    dsps_to_insts = defaultdict(list)
    dsp_input_regs_to_ssa = defaultdict(list)
    dsp_output_wire_to_ssa = defaultdict(list)
    dsp_counter = -1
    ssa_to_register = defaultdict(list)
    ssa_to_wire = {}

    output_to_ssa = {}
    stages = defaultdict(list)

    def task(stage_idx_stage):
        nonlocal dsp_counter

        stage_idx, stage = stage_idx_stage
        for var in stage:
            if var in defining_lines:
                defining_line = defining_lines[var]
                stages[stage_idx].append(defining_line)

                if "fmuladd" in defining_line:
                    dsp_counter += 1
                    dsp_counter %= MAX_DSPS

                    dsp = get_dsp(dsp_counter)
                    dsps_to_insts[dsp].append((stage_idx, defining_line))

                    assign, deps = get_ssas_from_ir_line(defining_line)
                    assert assign == var
                    # TODO: DOUBLE CHECK THE FMUL CONVENTION
                    # in reality i need a map from instruction to dsp based on the instruction
                    # blegh
                    dsp_input_regs_to_ssa[dsp.din0].append((stage_idx, deps[0]))
                    ssa_to_register[deps[0]].append((stage_idx, dsp.din0))
                    dsp_input_regs_to_ssa[dsp.din1].append((stage_idx, deps[1]))
                    ssa_to_register[deps[1]].append((stage_idx, dsp.din1))
                    dsp_input_regs_to_ssa[dsp.din2].append((stage_idx, deps[2]))
                    ssa_to_register[deps[2]].append((stage_idx, dsp.din2))

                    # TODO: what's the perf difference between what i have now and not doing this
                    # if this is the output of FMulAdd, it should be fed back in to the din2
                    dsp_output_wire_to_ssa[dsp.dout].append((stage_idx, var))
                    assert var not in ssa_to_wire
                    ssa_to_wire[var] = dsp.dout
                elif "store" in defining_line:
                    assign, deps = get_ssas_from_ir_line(defining_line)
                    output_to_ssa[assign] = deps[0]

        if stage_idx in stages:
            stages[stage_idx].sort()

    with ThreadPool(processes=NUM_THREADS) as pool:
        pool.map(task, enumerate(topo_sort_stages))

    print(
        dedent(
            """\
                `timescale 1ns/1ps

                module forward(
                    output ap_local_block,
                    output ap_local_deadlock,
                    input ap_clk,
                    input ap_rst,
                    input ap_start,
                    output ap_done,
                    output ap_idle,
                    output ap_ready,
            """
        ),
        file=FILE,
    )
    for inp in inputs:
        print(f"\tinput [31:0] {inp.replace('%', '')},", file=FILE)
    for i, outp in enumerate(outputs):
        print(f"\toutput [31:0] {outp.replace('%', '')},", file=FILE)
        print(
            f"\toutput {outp.replace('%', '')}_vld"
            + ("," if i < len(outputs) - 1 else ""),
            file=FILE,
        )
    print(");", file=FILE)

    total_num_stages = NUM_STAGES * len(stages)
    for i in range(total_num_stages):
        print(
            f"\tparameter ap_ST_fsm_state{i} = {total_num_stages}'d{1 << i};", file=FILE
        )

    print(file=FILE)

    print(
        indent(
            dedent(
                """\
                    reg ap_done;
                    reg ap_idle;
                    reg ap_ready;
                """
            ),
            "\t",
        ),
        file=FILE,
    )

    print(file=FILE)

    for i, outp in enumerate(outputs):
        print(f"\treg {outp.replace('%', '')}_vld;", file=FILE)

    print(
        indent(
            dedent(
                f"""\
                    (* fsm_encoding = "none" *) reg[{total_num_stages - 1}:0] ap_CS_fsm;
                    reg[{total_num_stages - 1}:0] ap_NS_fsm;
                """
            ),
            "\t",
        ),
        file=FILE,
    )
    for i in range(total_num_stages):
        print(f"\twire ap_CS_fsm_state{i};", file=FILE)

    print(file=FILE)

    print(f"\treg ap_ST_fsm_state0_blk;", file=FILE)
    for i in range(1, total_num_stages):
        print(f"\twire ap_ST_fsm_state{i}_blk;", file=FILE)

    print("\twire ap_ce_reg;", file=FILE)

    print(
        indent(
            dedent(
                f"""\
                
                    initial begin
                        #0 ap_CS_fsm = {total_num_stages - 1}'d1;
                    end
                    
                    always @(*) begin
                        if ((ap_start == 1'b0)) begin
                            ap_ST_fsm_state0_blk = 1'b1;
                        end else begin
                            ap_ST_fsm_state0_blk = 1'b0;
                        end
                    end
                """
            ),
            "\t",
        ),
        file=FILE,
    )

    print(file=FILE)

    for i in range(1, total_num_stages):
        print(f"\tassign ap_ST_fsm_state{i}_blk = 1'b0;", file=FILE)

    print(file=FILE)

    print(
        indent(
            dedent(
                f"""\
                    always @(*) begin
                        if ((1'b1 == ap_CS_fsm_state{total_num_stages - 1})) begin
                            ap_done = 1'b1;
                        end else begin
                            ap_done = 1'b0;
                        end
                    end

                    always @(*) begin
                        if (((1'b1 == ap_CS_fsm_state0) & (ap_start == 1'b0))) begin
                            ap_idle = 1'b1;
                        end else begin
                            ap_idle = 1'b0;
                        end
                    end

                    always @(*) begin
                        if ((1'b1 == ap_CS_fsm_state{total_num_stages - 1})) begin
                            ap_ready = 1'b1;
                        end else begin
                            ap_ready = 1'b0;
                        end
                    end
                    
                    always @(posedge ap_clk) begin
                        if (ap_rst == 1'b1) begin
                            ap_CS_fsm <= ap_ST_fsm_state0;
                        end else begin
                            ap_CS_fsm <= ap_NS_fsm;
                        end
                    end
                """
            ),
            "\t",
        ),
        file=FILE,
    )

    for i in range(len(dsps)):
        print(indent(mul_adder(i), "\t"), file=FILE)

    for ssa in ssa_to_register:
        if ssa in inputs or ssa in outputs:
            continue
        try:
            float(ssa)
            continue
        except:
            print(f"\treg   [31:0] {ssa.replace('%', '')};", file=FILE)

    for ssa in ssa_to_wire:
        if ssa in inputs or ssa in outputs:
            continue
        try:
            float(ssa)
            continue
        except:
            print(f"\treg   [31:0] {ssa.replace('%', '')};", file=FILE)

    print(file=FILE)

    # TODO: constants aren't formatted correctly

    for dsp in dsps.values():
        print("\talways @(*) begin", file=FILE)
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din0]):
            stage = NUM_STAGES * stage
            print(
                "\t\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin",
                file=FILE,
            )
            assign_stmt = " ".join(
                map(str, ["\t\t\t", dsp.din0, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt, file=FILE)
            print("\t\tend", file=FILE)
        # TODO:
        print(
            indent(
                dedent(
                    f"""\
                        else begin
                            {dsp.din0} = 'bx;
                        end
                    """
                ),
                "\t\t",
            ),
            file=FILE,
        )
        print("\tend\n", file=FILE)

        print(file=FILE)

        print("\talways @(*) begin", file=FILE)
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din1]):
            stage = NUM_STAGES * stage
            print(
                "\t\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin",
                file=FILE,
            )
            assign_stmt = " ".join(
                map(str, ["\t\t\t", dsp.din1, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt, file=FILE)
            print("\t\tend", file=FILE)
        print(
            indent(
                dedent(
                    f"""\
                        else begin
                            {dsp.din1} = 'bx;
                        end
                    """
                ),
                "\t\t",
            ),
            file=FILE,
        )
        print("\tend\n", file=FILE)

        print(file=FILE)

        print("\talways @(*) begin", file=FILE)
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din2]):
            stage = NUM_STAGES * stage
            print(
                "\t\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin",
                file=FILE,
            )
            assign_stmt = " ".join(
                map(str, ["\t\t\t", dsp.din2, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt, file=FILE)
            print("\t\tend", file=FILE)
        print(
            indent(
                dedent(
                    f"""\
                        else begin
                            {dsp.din2} = 'bx;
                        end
                    """
                ),
                "\t\t",
            ),
            file=FILE,
        )
        print("\tend\n", file=FILE)

    verilog_stage_outputs = defaultdict(list)

    for dsp in dsps.values():
        for i, (stage, ssa) in enumerate(dsp_output_wire_to_ssa[dsp.dout]):
            stage = NUM_STAGES * stage
            assign_stmt = " ".join(
                map(str, [ssa.replace("%", ""), "<=", f"{dsp.dout};"])
            )
            verilog_stage_outputs[stage + 1].append(assign_stmt)

    for stage in verilog_stage_outputs:
        print("\talways @ (posedge ap_clk) begin", file=FILE)
        print(f"\t\tif ((1'b1 == ap_CS_fsm_state{stage})) begin", file=FILE)
        for output in verilog_stage_outputs[stage]:
            print("\t\t\t" + output, file=FILE)

        print("\t\tend", file=FILE)
        print("\tend", file=FILE)

    for outp in outputs:
        print(
            indent(
                dedent(
                    f"""\
                        always @(*) begin
                            if ((1'b1 == ap_CS_fsm_state{total_num_stages - 1})) begin
                                {outp.replace('%', '')}_vld = 1'b1;
                            end else begin
                                {outp.replace('%', '')}_vld = 1'b0;
                            end
                        end
                    """
                ),
                "\t",
            ),
            file=FILE,
        )

    print(
        indent(
            dedent(
                """\
                    always @(*) begin
                        case (ap_CS_fsm)
                            ap_ST_fsm_state1: begin
                                if (((1'b1 == ap_CS_fsm_state0) & (ap_start == 1'b1))) begin
                                    ap_NS_fsm = ap_ST_fsm_state1;
                                end else begin
                                    ap_NS_fsm = ap_ST_fsm_state0;
                                end
                            end
                """
            ),
            "\t",
        ),
        file=FILE,
    )
    for i in range(1, total_num_stages - 1):
        print(
            indent(
                dedent(
                    f"""\
                        ap_ST_fsm_state{i}: begin
                            ap_NS_fsm = ap_ST_fsm_state{i + 1};
                        end
                    """
                ),
                "\t\t\t",
            ),
            file=FILE,
        )

    print(
        indent(
            dedent(
                f"""\
                            ap_ST_fsm_state{total_num_stages - 1}: begin
                                ap_NS_fsm = ap_ST_fsm_state0;
                            end
                            default: begin
                                ap_NS_fsm = 'bx;
                            end
                        endcase
                    end
                """
            ),
            "\t",
        ),
        file=FILE,
    )

    for i in range(total_num_stages - 1):
        print(
            indent(
                dedent(
                    f"""\
                        assign ap_CS_fsm_state{i} = ap_CS_fsm[32'd{i}];
                    """
                ),
                "\t",
            ),
            file=FILE,
        )

    print(
        indent(
            dedent(
                """\
                    assign ap_local_block = 1'b0;
                    assign ap_local_deadlock = 1'b0;
                """
            ),
            "\t",
        ),
        file=FILE,
    )

    for outp in outputs:
        print(
            indent(
                dedent(
                    f"""\
                        assign {outp.replace('%', '')} = {ssa_to_wire[output_to_ssa[outp]]};
                    """
                ),
                "\t",
            ),
            file=FILE,
        )

    print("endmodule", file=FILE)


def main():
    parser = argparse.ArgumentParser(description="make stuff")
    parser.add_argument("fp", type=Path)
    args = parser.parse_args()

    lines = open(args.fp).readlines()

    all_vals, G, Ginv, defining_lines, inputs, outputs = parallel_make_graph(lines)
    stages = parallel_topo_sort(all_vals, G, Ginv)
    make_schedule(stages, defining_lines, inputs, outputs)


if __name__ == "__main__":
    main()
