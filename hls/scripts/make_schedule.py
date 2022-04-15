import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

from toposort import toposort

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


def make_schedule(fp):
    lines = open(fp).readlines()

    G = defaultdict(set)
    defining_lines = {}
    inputs = []
    outputs = []

    for i, line in enumerate(lines):
        assign, deps = get_ssas_from_ir_line(line)
        if assign is None:
            continue

        if "define" in line:
            inputs = assign
            outputs = deps
            continue
        if "declare" in line:
            continue

        defining_lines[assign] = line

        for dep in deps:
            G[assign].add(dep)

    stages = defaultdict(list)

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

    topo_sort_stages = {}
    for i, stage in enumerate(toposort(G)):
        topo_sort_stages[i] = stage
        for var in stage:
            if var in defining_lines:
                defining_line = defining_lines[var]
                stages[i].append(defining_line)

                if "fmuladd" in defining_line:
                    dsp_counter += 1
                    dsp_counter %= MAX_DSPS

                    dsp = get_dsp(dsp_counter)
                    dsps_to_insts[dsp].append((i, defining_line))

                    assign, deps = get_ssas_from_ir_line(defining_line)
                    assert assign == var
                    # TODO: DOUBLE CHECK THE FMUL CONVENTION
                    # in reality i need a map from instruction to dsp based on the instruction
                    # blegh
                    dsp_input_regs_to_ssa[dsp.din0].append((i, deps[0]))
                    ssa_to_register[deps[0]].append((i, dsp.din0))
                    dsp_input_regs_to_ssa[dsp.din1].append((i, deps[1]))
                    ssa_to_register[deps[1]].append((i, dsp.din1))
                    dsp_input_regs_to_ssa[dsp.din2].append((i, deps[2]))
                    ssa_to_register[deps[2]].append((i, dsp.din2))

                    # TODO: what's the perf difference between what i have now and not doing this
                    # if this is the output of FMulAdd, it should be fed back in to the din2
                    dsp_output_wire_to_ssa[dsp.dout].append((i, var))
                    assert var not in ssa_to_wire
                    ssa_to_wire[var] = dsp.dout
                elif "store" in defining_line:
                    assign, deps = get_ssas_from_ir_line(defining_line)
                    output_to_ssa[assign] = deps[0]

        if i in stages:
            stages[i].sort()

    print(
        """
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
    )
    for inp in inputs:
        print(f"\tinput [31:0] {inp.replace('%', '')},")
    for i, outp in enumerate(outputs):
        print(f"\toutput [31:0] {outp.replace('%', '')},")
        print(f"\toutput {outp.replace('%', '')}_vld" + ("," if i < len(outputs) - 1 else ""))
    print(");")

    total_num_stages = NUM_STAGES * len(stages)
    for i in range(total_num_stages):
        print(f"    parameter ap_ST_fsm_state{i} = {total_num_stages}'d{1 << i};")

    print()

    print(
        """
    reg ap_done;
    reg ap_idle;
    reg ap_ready;
    """
    )

    print()

    for i, outp in enumerate(outputs):
        print(f"\treg {outp.replace('%', '')}_vld;")

    print(
        f"""
    (* fsm_encoding = "none" *) reg[{total_num_stages - 1}:0] ap_CS_fsm;
    reg[{total_num_stages - 1}:0] ap_NS_fsm;
    """
    )
    for i in range(total_num_stages):
        print(f"\twire ap_CS_fsm_state{i};")

    print()

    print(f"\treg ap_ST_fsm_state0_blk;")
    for i in range(1, total_num_stages):
        print(f"\twire ap_ST_fsm_state{i}_blk;")

    print("\twire ap_ce_reg;")

    print(
        f"""
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
    )

    print()

    for i in range(1, total_num_stages):
        print(f"\tassign ap_ST_fsm_state{i}_blk = 1'b0;")

    print()

    print(
        f"""
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
    )

    for i in range(len(dsps)):
        print(mul_adder(i))

    for ssa in ssa_to_register:
        if ssa in inputs or ssa in outputs:
            continue
        try:
            float(ssa)
            continue
        except:
            print(f"reg   [31:0] {ssa.replace('%', '')};")

    for ssa in ssa_to_wire:
        if ssa in inputs or ssa in outputs:
            continue
        try:
            float(ssa)
            continue
        except:
            print(f"reg   [31:0] {ssa.replace('%', '')};")

    print()

    # TODO: constants aren't formatted correctly

    for dsp in dsps.values():
        print("always @(*) begin")
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din0]):
            stage = NUM_STAGES * stage
            print(
                "\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin"
            )
            assign_stmt = " ".join(
                map(str, ["\t\t", dsp.din0, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt)
            print("\tend")
        # TODO:
        print(
            f"""\telse begin
        {dsp.din0} = 'bx;
    end
        """
        )
        print("end\n")

        print()

        print("always @(*) begin")
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din1]):
            stage = NUM_STAGES * stage
            print(
                "\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin"
            )
            assign_stmt = " ".join(
                map(str, ["\t\t", dsp.din1, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt)
            print("\tend")
        print(
            f"""\telse begin
        {dsp.din1} = 'bx;
    end
        """
        )
        print("end\n")

        print()

        print("always @(*) begin")
        for i, (stage, ssa) in enumerate(dsp_input_regs_to_ssa[dsp.din2]):
            stage = NUM_STAGES * stage
            print(
                "\t"
                + ("else " if i > 0 else "")
                + f"if ((1'b1 == ap_CS_fsm_state{stage})) begin"
            )
            assign_stmt = " ".join(
                map(str, ["\t\t", dsp.din2, "=", ssa.replace("%", ""), ";"])
            )
            print(assign_stmt)
            print("\tend")
        print(
            f"""\telse begin
        {dsp.din2} = 'bx;
    end
        """
        )
        print("end\n")

    verilog_stage_outputs = defaultdict(list)

    for dsp in dsps.values():
        for i, (stage, ssa) in enumerate(dsp_output_wire_to_ssa[dsp.dout]):
            stage = NUM_STAGES * stage
            assign_stmt = " ".join(
                map(str, ["\t", ssa.replace("%", ""), "<=", dsp.dout, ";"])
            )
            verilog_stage_outputs[stage + 1].append(assign_stmt)

    for stage in verilog_stage_outputs:
        print("always @ (posedge ap_clk) begin")
        print(f"if ((1'b1 == ap_CS_fsm_state{stage})) begin")
        for output in verilog_stage_outputs[stage]:
            print("\t" + output)

        print("\tend")
        print("end")

    for outp in outputs:
        print(
            f"""
    always @(*) begin
        if ((1'b1 == ap_CS_fsm_state{total_num_stages - 1})) begin
            {outp.replace('%', '')}_vld = 1'b1;
        end else begin
            {outp.replace('%', '')}_vld = 1'b0;
        end
    end
        """
        )

    print(
        """
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
    )
    for i in range(1, total_num_stages - 1):
        print(
            f"""
            ap_ST_fsm_state{i}: begin
                ap_NS_fsm = ap_ST_fsm_state{i + 1};
            end
        """
        )

    print(
        f"""
            ap_ST_fsm_state{total_num_stages - 1}: begin
                ap_NS_fsm = ap_ST_fsm_state0;
            end
            default: begin
                ap_NS_fsm = 'bx;
            end
        endcase
    end
    """
    )

    for i in range(total_num_stages - 1):
        print(
            f"""
    assign ap_CS_fsm_state{i} = ap_CS_fsm[32'd{i}];
        """
        )

    print(
        """
    assign ap_local_block = 1'b0;
    assign ap_local_deadlock = 1'b0;
    """
    )

    for outp in outputs:
        print(
            f"""
    assign {outp.replace('%', '')} = {ssa_to_wire[output_to_ssa[outp]]};
        """
        )

    print("endmodule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make stuff")
    parser.add_argument("fp", type=Path)
    args = parser.parse_args()
    make_schedule(str(args.fp))
