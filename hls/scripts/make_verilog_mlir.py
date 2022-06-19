import argparse
import enum
import json
import math
import os
import random
import struct
import sys
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import List, Dict, Tuple

import networkx as nx
import numpy as np


ParsedProgram = namedtuple(
    "ParsedProgram",
    ["inputs", "outputs", "pes_to_vals", "ops_to_pes", "constants", "max_interval"],
)


def format_cst(cst):
    return np.random.randint(1, 100000)
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


def float_to_int(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]


def float_to_hex(f):
    return hex(float_to_int(f))


def half_to_int(f):
    return struct.unpack("<H", struct.pack("<e", f))[0]


def half_to_hex(f):
    return hex(half_to_int(f))


def make_fsm_states(fsm_states, fsm_idx_width):
    s = " | ".join(
        [
            f"(1'b1 == current_state_fsm_state{str(i).zfill(fsm_idx_width)})"
            for i in fsm_states
        ]
    )
    return s


def make_var_val_reg(idx):
    return f"arr{'_'.join(idx)}"


class RegOrWire(enum.Enum):
    REG = "reg"
    WIRE = "wire"


@dataclass(frozen=True)
class Val:
    reg_or_wire: enum.Enum
    id: str

    @property
    def name(self):
        return f"{self.id}_{self.reg_or_wire.value}"

    def __str__(self):
        return self.name


class Op(enum.Enum):
    MUL = "mul"
    ADD = "add"
    RELU = "relu"


COLLAPSE_TREES = True
MAX_FANOUT = 100
USE_BRAM = False
PRECISION = 11
LAYERS = 1
KEEP = False


def make_mul_or_add(precision, id, op_name, a_reg, b_reg, res_wire, add_or_mul):
    if precision < 16:
        op = dedent(
            f"""\
                wire   [{precision - 1}:0] {res_wire};
                {'(* keep = "true" *)' if KEEP else ''} f{add_or_mul} #({id}, {precision}) f{op_name}(
                    .clk(clock),
                    .rst(0'b1),
                    .X({a_reg}),
                    .Y({b_reg}),
                    .R({res_wire})
                );
                """
        )
    else:
        op = dedent(
            f"""\
                wire   [{precision - 1}:0] {res_wire};
                {'(* keep = "true" *)' if KEEP else ''} f{add_or_mul} #({id}, {precision}) f{op_name}(
                    .clock(clock),
                    .reset(0'b1),
                    .clock_enable(1'b1),
                    .a({a_reg}),
                    .b({b_reg}),
                    .res({res_wire})
                );
                """
        )
    return op


def make_relu_or_neg(precision, id, op_name, a_reg, res_wire, relu_or_neg):
    op = dedent(
        f"""\
            wire   [{precision - 1}:0] {res_wire};
            {relu_or_neg} #({id}, {precision}) {op_name}(
                .a({a_reg}),
                .res({res_wire})
            );
            """
    )
    return op


def make_shift_rom(precision, id, op_name, data_out_wire, addr_width):
    res = []
    # waddr = ','.join([f"0'b0"] * addr_width)
    # write_data = ','.join([f"0'b0"] * precision)
    res.append(
        dedent(
            f"""\
            reg[{addr_width}-1:0] {op_name}_raddr;
            always @(posedge clock) begin
                if ({op_name}_raddr == RAM_SIZE) begin
                    {op_name}_raddr = 0;
                end else begin
                    {op_name}_raddr <= {op_name}_raddr+1'b1;
                end
            end
        
            wire   [{precision - 1}:0] {data_out_wire};
            simple_dual_rw_ram #({id}, {precision}, {2 ** addr_width}) {op_name}(
                .wclk(clock),
                .waddr({addr_width}'b0),
                .write_data({op_name}_raddr),
                .write_en(1'b1),
                .rclk(clock),
                .raddr({op_name}_raddr),
                .read_data({data_out_wire})
            );
            """
        )
    )
    return "\n".join(res)


PE_IDXS_TO_IDS = {}


class FAddOrMulOp:
    def __init__(self, idx, precision, add_or_mul) -> None:
        global PE_IDXS_TO_IDS
        if not isinstance(idx, (tuple, list)):
            idx = tuple([idx])
        if idx not in PE_IDXS_TO_IDS:
            PE_IDXS_TO_IDS[idx] = len(PE_IDXS_TO_IDS)
        self.id = PE_IDXS_TO_IDS[idx]
        self.idx_str = "_".join(map(str, idx))
        self.add_or_mul = add_or_mul
        self.precision = precision
        self.a_reg = Val(RegOrWire.REG, f"f{self.name}_a")
        self.b_reg = Val(RegOrWire.REG, f"f{self.name}_b")
        self.res_reg = Val(RegOrWire.REG, f"f{self.name}_res")
        self.res_wire = Val(RegOrWire.WIRE, f"f{self.name}_res")
        self.registers = [self.a_reg, self.b_reg, self.res_reg]

    @property
    def name(self):
        return f"{self.add_or_mul}_{self.idx_str}"

    def pos_to_reg(self, pos):
        if pos == 0:
            return self.a_reg
        elif pos == 1:
            return self.b_reg
        else:
            raise Exception(f"unknown pos {pos}")

    def __str__(self):
        return self.name

    def make(self):
        return make_mul_or_add(
            self.precision,
            self.id,
            self.name,
            self.a_reg,
            self.b_reg,
            self.res_wire,
            self.add_or_mul,
        )


class FAdd(FAddOrMulOp):
    def __init__(self, idx, precision) -> None:
        super().__init__(idx, precision, "add")


class FMul(FAddOrMulOp):
    def __init__(self, idx, precision) -> None:
        super().__init__(idx, precision, "mul")


class ShiftROM:
    def __init__(self, idx, precision, num_csts):
        global PE_IDXS_TO_IDS
        if not isinstance(idx, (tuple, list)):
            idx = tuple([idx])
        if idx not in PE_IDXS_TO_IDS:
            PE_IDXS_TO_IDS[idx] = len(PE_IDXS_TO_IDS)
        self.id = PE_IDXS_TO_IDS[idx]
        self.idx_str = "_".join(map(str, idx))
        self.precision = precision
        self.data_out_wire = Val(RegOrWire.WIRE, f"{self.name}_data_out")
        self.num_csts = num_csts
        self.addr_width = math.ceil(math.log2(num_csts))

    @property
    def name(self):
        return f"shift_rom_{self.idx_str}"

    def make(self):
        return make_shift_rom(
            self.precision, self.id, self.name, self.data_out_wire, self.addr_width
        )


class ReLUOrNeg:
    def __init__(self, idx, precision, relu_or_neg) -> None:
        global PE_IDXS_TO_IDS
        if not isinstance(idx, (tuple, list)):
            idx = tuple([idx])
        if idx not in PE_IDXS_TO_IDS:
            PE_IDXS_TO_IDS[idx] = len(PE_IDXS_TO_IDS)
        self.id = PE_IDXS_TO_IDS[idx]
        self.idx_str = "_".join(map(str, idx))
        self.relu_or_neg = relu_or_neg
        self.precision = precision
        self.a_reg = Val(RegOrWire.REG, f"{relu_or_neg}_{self.idx_str}_a")
        self.res_wire = Val(RegOrWire.WIRE, f"{relu_or_neg}_{self.idx_str}_res")
        self.registers = [self.a_reg]

    @property
    def name(self):
        return f"{self.relu_or_neg}_{self.idx_str}"

    def make(self):
        return make_relu_or_neg(
            self.precision,
            self.id,
            self.name,
            self.a_reg,
            self.res_wire,
            self.relu_or_neg,
        )


class ReLU(ReLUOrNeg):
    def __init__(self, idx, precision) -> None:
        super().__init__(idx, precision, "relu")


class Neg(ReLUOrNeg):
    def __init__(self, idx, precision) -> None:
        super().__init__(idx, precision, "neg")


def make_always_tree(left, rights, comb_or_seq, fsm_idx_width):
    always_a = dedent(
        f"""\
        always @ ({'*' if comb_or_seq == 'comb' else 'posedge clock'}) begin
        """
    )
    if COLLAPSE_TREES:
        _rights = defaultdict(list)
        for stage, inp in rights:
            _rights[inp].append(stage)

        for inp in _rights:
            _rights[inp] = sorted(_rights[inp])
        rights = [(stages, inp) for inp, stages in _rights.items()]

    rights = sorted(rights, key=lambda x: len(x[0]))
    for i, (stage, inp_a) in enumerate(rights):
        if len(rights) == 1:
            cond = "begin"
        elif i == 0:
            cond = f"if ({make_fsm_states(stage, fsm_idx_width)}) begin"
        # else:
        elif i < len(rights) - 1:
            cond = f"else if ({make_fsm_states(stage, fsm_idx_width)}) begin"
        else:
            cond = f"else begin"

        always_a += indent(
            dedent(
                f"""\
                    {cond} // num states: {len(stage)}
                        {left} = {inp_a};
                    end
                """
            ),
            "\t",
        )
    always_a += "end"
    return always_a


class Module:
    def __init__(
        self, parsed_program: ParsedProgram, program_graph, op_latencies, precision=16
    ):
        self.precision = precision
        self.rtl_graph = nx.MultiDiGraph()
        self.mul_instances: Dict[Tuple[int, int], FMul] = {}
        self.add_instances: Dict[Tuple[int, int], FAdd] = {}
        self.relu_instances: Dict[int, ReLU] = {}
        self.neg_instances: Dict[int, Neg] = {}
        if USE_BRAM:
            self.rom_instances: Dict[Tuple[int, int], ShiftROM] = {}

        self.name_to_val = {}
        self.register_ids = set()
        self.wire_ids = set()

        self.program_val_to_rtl_reg = {}
        self.input_wires = [Val(RegOrWire.WIRE, v) for v in parsed_program.inputs]
        self.output_wires = [Val(RegOrWire.WIRE, v) for v in parsed_program.outputs]
        self.input_regs = {v: Val(RegOrWire.REG, v) for v in parsed_program.inputs}
        self.program_val_to_rtl_reg.update(self.input_regs)
        self.add_vals_to_rtl_graph(
            self.input_wires + self.output_wires + list(self.input_regs.values())
        )
        self.constants = parsed_program.constants
        for cst in self.constants:
            v = self.program_val_to_rtl_reg[cst] = Val(RegOrWire.REG, cst)
            self.add_vals_to_rtl_graph([v])

        for pe_idx in parsed_program.ops_to_pes["fmul"]:
            mul = FMul(pe_idx, precision)
            self.mul_instances[pe_idx] = mul
            self.add_vals_to_rtl_graph(mul.registers)
            self.add_vals_to_rtl_graph([mul.res_wire])

            if USE_BRAM:
                rom = ShiftROM(pe_idx, precision, num_csts=512)
                self.rom_instances[pe_idx] = rom
                self.add_vals_to_rtl_graph([rom.data_out_wire])

        for pe_idx in parsed_program.ops_to_pes["fadd"]:
            add = FAdd(pe_idx, precision)
            self.add_instances[pe_idx] = add
            self.add_vals_to_rtl_graph(add.registers)
            self.add_vals_to_rtl_graph([add.res_wire])

        for pe_idx in parsed_program.ops_to_pes.get("relu", []):
            relu = ReLU(pe_idx, precision)
            self.relu_instances[pe_idx] = relu
            self.add_vals_to_rtl_graph([relu.a_reg, relu.res_wire])

        for pe_idx in parsed_program.ops_to_pes.get("fneg", []):
            neg = Neg(pe_idx, precision)
            self.neg_instances[pe_idx] = neg
            self.add_vals_to_rtl_graph([neg.a_reg, neg.res_wire])

        self.max_fsm_stage = parsed_program.max_interval

        def add_add_mul_transfer(stage, op_idx, op_type, inp_pos, inp):
            op = (
                self.mul_instances[op_idx]
                if op_type == Op.MUL
                else self.add_instances[op_idx]
            )

            if inp == op.res_reg:
                warnings.warn(f"{stage} {op_idx} {op_type} {inp_pos} {inp} {op.res_reg} comb loop")

            if inp_pos == 0:
                assert inp != op.a_reg
                self.rtl_graph.add_edge(inp, op.a_reg, stage=stage)
            elif inp_pos == 1:
                assert inp != op.b_reg
                self.rtl_graph.add_edge(inp, op.b_reg, stage=stage)
            else:
                raise Exception(f"unknown pos {pos}")
            return op.res_reg

        for u in nx.topological_sort(program_graph):
            if u == "return":
                continue
            assert u in self.program_val_to_rtl_reg, f"unknown u {u}"
            src = self.program_val_to_rtl_reg[u]
            for v, attrs in program_graph[u].items():
                for attr in attrs.values():
                    op_info = program_graph.nodes[v]
                    op = op_info["op"]
                    if op == "return":
                        # TODO
                        continue
                    pos = attr["pos"]
                    pe_idx = op_info["pe_idx"]
                    stage = op_info["start_time"]
                    if op in {"fmul", "fadd"}:
                        op_type = Op.MUL if op == "fmul" else Op.ADD
                        res = add_add_mul_transfer(
                            stage, pe_idx, op_type, inp_pos=pos, inp=src
                        )
                    elif op in {"relu", "fneg"}:
                        if op == "relu":
                            op = self.relu_instances[pe_idx]
                        else:
                            op = self.neg_instances[pe_idx]
                        self.rtl_graph.add_edge(src, op.a_reg, stage=stage)
                        res = op.res_wire
                    else:
                        raise Exception(f"unknown op {op}")

                    if v not in self.program_val_to_rtl_reg:
                        self.program_val_to_rtl_reg[v] = res

    @property
    def fsm_idx_width(self):
        return self._fsm_idx_width

    @property
    def max_fsm_stage(self):
        return self._max_fsm_stage

    @max_fsm_stage.setter
    def max_fsm_stage(self, value):
        self._max_fsm_stage = value
        self._fsm_idx_width = math.ceil(math.log10(self.max_fsm_stage))

    def add_vals_to_rtl_graph(self, vals: List[Val]):
        assert all(isinstance(v, Val) for v in vals)
        self.rtl_graph.add_nodes_from(vals)
        for r in vals:
            self.name_to_val[r.name] = r
            if r.reg_or_wire == RegOrWire.REG:
                self.register_ids.add(r.id)
            if r.reg_or_wire == RegOrWire.WIRE:
                self.wire_ids.add(r.id)

    def make_assign_wire_to_reg(self):
        assigns = []
        for wire_reg in self.wire_ids.intersection(self.register_ids):
            if USE_BRAM and "constant" in wire_reg:
                continue

            assigns.append(
                dedent(
                    f"""\
            always @ (posedge clock) begin
                {wire_reg}_reg <= {wire_reg}_wire;
            end
            """
                )
            )

        return "\n".join(assigns)

    def make_top_module_decl(self):
        inputs = self.input_wires
        outputs = self.output_wires
        base_inputs = ["clock", "reset", "clock_enable"]
        input_ports = [f"[{self.precision - 1}:0] {i.name}" for i in inputs]

        base_outputs = []
        output_ports = [f"[{self.precision - 1}:0] {o.name}" for o in outputs]

        input_wires = ",\n".join(
            [f"input wire {inp}" for inp in base_inputs + input_ports]
        )
        output_wires = ",\n".join(
            [f"output wire {outp}" for outp in base_outputs + output_ports]
        )

        mod_top = dedent(
            f"""\
        `default_nettype none
        module forward (
        """
        )

        mod_top += indent(
            dedent(
                ",\n".join(
                    [f"input wire {inp}" for inp in base_inputs + input_ports[:2]]
                )
            ),
            "\t",
        )
        mod_top += ",\n"
        mod_top += indent(
            dedent(",\n".join([f"output wire {inp}" for inp in output_ports])), "\t"
        )
        mod_top += "\n);\n\n"
        mod_top += dedent(
            "\n".join(
                [
                    f"""reg {inp} = {self.precision}'d{random.randint(0, 2 ** self.precision - 1)};"""
                    for inp in input_ports[2:]
                ]
            )
        )

        mod_top += "\n"
        mod_top += "\nforward_inner #(512) _forward_inner(\n"
        mod_top += indent(
            dedent(
                ",\n".join(
                    [f".{port}({port})" for port in base_inputs + inputs + outputs]
                )
            ),
            "\t",
        )
        mod_top += "\n"
        mod_top += dedent(
            f"""\
        );
        endmodule
        """
        )

        mod_inner = dedent(
            f"""\
        module forward_inner #(RAM_SIZE = 512) (
        """
        )
        mod_inner += indent(dedent(input_wires + ",\n" + output_wires), "\t")
        mod_inner += "\n);\n"

        return "\n".join([mod_top, mod_inner])

    def make_fsm_params(self):
        params = "\n".join(
            [
                f"parameter fsm_state{str(i).zfill(self.fsm_idx_width)} = {self.max_fsm_stage + 1}'d{1 << i - 1};"
                for i in range(1, self.max_fsm_stage + 1)
            ]
        )
        params += "\n\n"
        params += f'(* max_fanout = {MAX_FANOUT}, fsm_encoding = "none" *) reg [{self.max_fsm_stage}:0] current_state_fsm;\n'
        params += f"reg [{self.max_fsm_stage}:0] next_state_fsm;"

        return params

    def make_fsm_wires(self):
        wires = "\n".join(
            [
                f"wire current_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(1, self.max_fsm_stage + 1)
            ]
        )
        return wires

    def make_fsm(self):
        first_state = str(1).zfill(self.fsm_idx_width)
        fsm = dedent(
            f"""\
            always @ (posedge clock) begin
                if (reset == 1'b1) begin
                    current_state_fsm <= fsm_state{first_state};
                end else begin
                    current_state_fsm <= next_state_fsm;
                end
            end

            always @ (*) begin
                case (current_state_fsm)
            """
        )
        for i in range(1, self.max_fsm_stage):
            fsm += indent(
                dedent(
                    f"""\
                fsm_state{str(i).zfill(self.fsm_idx_width)} : begin
                    next_state_fsm = fsm_state{str(i + 1).zfill(self.fsm_idx_width)};
                end
                """
                ),
                "\t\t",
            )

        fsm += indent(
            dedent(
                f"""\
            fsm_state{str(self.max_fsm_stage).zfill(self.fsm_idx_width)} : begin
                next_state_fsm = fsm_state{str(1).zfill(self.fsm_idx_width)};
            end
            """
            ),
            "\t\t",
        )

        fsm += indent(
            dedent(
                """\
        default : begin
            next_state_fsm = 'bx;
        end
        """
            ),
            "\t\t",
        )
        fsm += dedent(
            """\
            endcase
        end
        """
        )

        fsm_2bit_width = math.ceil(math.log2(self.max_fsm_stage))
        for i in range(1, self.max_fsm_stage):
            fsm += dedent(
                f"""\
        assign current_state_fsm_state{str(i).zfill(self.fsm_idx_width)} = current_state_fsm[{fsm_2bit_width}'d{i - 1}];
            """
            )

        return fsm

    def make_trees(self):
        adj = self.rtl_graph.reverse().adj
        for val in self.rtl_graph.nodes:
            edges = adj[val]
            if edges:
                if "fmul" in val.name and "_b" in val.name:
                    arms = sorted(
                        [
                            (edge_attr["stage"], v.name)
                            for v, edge_attrs in edges.items()
                            for _, edge_attr in edge_attrs.items()
                            if not USE_BRAM or "constant" not in v.name
                        ]
                    )
                else:
                    arms = sorted(
                        [
                            (edge_attr["stage"], v.name)
                            for v, edge_attrs in edges.items()
                            for _, edge_attr in edge_attrs.items()
                        ]
                    )
                if arms:
                    comb_or_seq = "comb"
                    tree = make_always_tree(
                        val.name,
                        arms,
                        comb_or_seq=comb_or_seq,
                        fsm_idx_width=self.fsm_idx_width,
                    )
                    yield tree
                else:
                    yield ""

    def make_op_instances(self):
        for mul in self.mul_instances.values():
            yield mul.make()
        for add in self.add_instances.values():
            yield add.make()
        for relu in self.relu_instances.values():
            yield relu.make()
        for neg in self.neg_instances.values():
            yield neg.make()
        if USE_BRAM:
            for rom in self.rom_instances.values():
                yield rom.make()

    def make_registers(self):
        res = []
        for reg in self.rtl_graph:
            if reg.reg_or_wire == RegOrWire.WIRE:
                continue
            if "constant" in reg.name:
                res.extend(
                    [
                        f"(* max_fanout = {MAX_FANOUT} *) reg [{self.precision - 1}:0] {reg.name};",
                        f"initial {reg.name} = 16'd{random.randint(0, 2 ** self.precision - 1)};",
                    ]
                )
            else:
                res.append(
                    f"(* max_fanout = {MAX_FANOUT} *) reg [{self.precision - 1}:0] {reg.name};"
                )

        return "\n".join(res)

    def make_outputs(self, program_graph):
        rets = []
        rev = program_graph.reverse()
        for output in self.output_wires:
            last_val = list(rev[output.id].keys())[0]
            last_reg = self.program_val_to_rtl_reg[last_val]
            assert last_reg.id in self.register_ids, last_reg
            rets.append(f"assign {output} = {last_reg};")
        return "\n".join(rets)


def parse_program_graph(G):
    inputs = {}
    vals_to_pes = {}
    pes_to_vals = defaultdict(list)
    ops_to_pes = defaultdict(set)
    constants = {}
    for n, attrdict in G.nodes.items():
        op = attrdict["op"]
        if op == "arg":
            inputs[n] = attrdict
        elif op == "return":
            continue
        elif op == "constant":
            constants[n] = attrdict["literal"]
        else:
            pe_idx = attrdict["pe_idx"]
            assert pe_idx is not None
            pe_idx = tuple(pe_idx)
            attrdict["pe_idx"] = pe_idx
            vals_to_pes[n] = pe_idx
            pes_to_vals[pe_idx].append(n)
            ops_to_pes[op].add(pe_idx)

    outputs = G.reverse().adj["return"].copy()
    max_interval = G.nodes["return"]["start_time"]

    return ParsedProgram(
        inputs, outputs, pes_to_vals, ops_to_pes, constants, max_interval
    )


def main(design_fp, precision):
    design = json.load(open(design_fp))
    program_graph = nx.json_graph.node_link_graph(design["program_graph"])
    parsed_prog = parse_program_graph(program_graph)
    op_latencies = design["op_latencies"]

    mod = Module(parsed_prog, program_graph, op_latencies, precision=precision)

    verilog_fp = os.path.split(design_fp)[0]
    verilog_file = open(f"{verilog_fp}/forward.v", "w")

    print(mod.make_top_module_decl(), file=verilog_file)
    print(file=verilog_file)
    print(mod.make_fsm_params(), file=verilog_file)
    print(file=verilog_file)
    print(mod.make_fsm_wires(), file=verilog_file)
    print(file=verilog_file)
    print(mod.make_registers(), file=verilog_file)
    print(file=verilog_file)
    for inst in mod.make_op_instances():
        print(inst, file=verilog_file)
    print(file=verilog_file)
    print(mod.make_assign_wire_to_reg(), file=verilog_file)
    print(file=verilog_file)
    for tree in mod.make_trees():
        print(tree, file=verilog_file)
    print(file=verilog_file)
    print(mod.make_fsm(), file=verilog_file)
    print(file=verilog_file)
    print(mod.make_outputs(program_graph), file=verilog_file)
    print(file=verilog_file)
    print("endmodule", file=verilog_file)

    settings_file = open(f"{verilog_fp}/settings.txt", "w")
    settings_file.write(
        json.dumps(
            {
                "COLLAPSE_TREES": COLLAPSE_TREES,
                "MAX_FANOUT": MAX_FANOUT,
                "USE_BRAM": USE_BRAM,
                "PRECISION": PRECISION,
                "KEEP": KEEP,
                "NUM_PES": len(PE_IDXS_TO_IDS)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("design_fp")
    parser.add_argument("--precision", default=11, type=int)
    args = parser.parse_args()
    main(args.design_fp, args.precision)
