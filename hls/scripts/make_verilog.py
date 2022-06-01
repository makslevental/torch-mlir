import enum
import json
import math
import struct
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from textwrap import dedent, indent
from typing import List, Dict

import networkx as nx
import numpy as np


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


INTERVAL_BETWEEN_MUL_AND_ADD = 1
MUL_LATENCY = 1
ADD_LATENCY = 1
FSM_STAGE_INTERVAL = 2


def latency_for_op(op):
    if "fmuladd" in op:
        return MUL_LATENCY + 1 + ADD_LATENCY + 1
    if "fmul" in op:
        return MUL_LATENCY + 1
    if "fadd" in op:
        return ADD_LATENCY + 1
    raise Exception(f"unknown op {op}")


def max_latency_per_fsm_stage(fsm_stages):
    stage_latencies = {}
    for i, stage in fsm_stages.items():
        stage_latencies[i] = max([latency_for_op(op) for op in stage])

    return stage_latencies


def make_mul_or_add(precision, idx, op_name, a_reg, b_reg, res_wire, add_or_mul):
    op = dedent(
        f"""\
            wire   [{precision - 1}:0] {res_wire};
            f{add_or_mul} #({idx}) f{op_name}(
                .clock(clock),
                .reset(reset),
                .clock_enable(clock_enable),
                .a({a_reg}),
                .b({b_reg}),
                .res({res_wire})
            );
            """
    )
    return op


class FAddOrMulOp:
    def __init__(self, idx, precision, add_or_mul) -> None:
        self.idx = idx
        self.add_or_mul = add_or_mul
        self.precision = precision
        self.a_reg = Val(RegOrWire.REG, f"f{add_or_mul}_{idx}_a")
        self.b_reg = Val(RegOrWire.REG, f"f{add_or_mul}_{idx}_b")
        self.res_reg = Val(RegOrWire.REG, f"f{add_or_mul}_{idx}_res")
        self.res_wire = Val(RegOrWire.WIRE, f"f{add_or_mul}_{idx}_res")
        self.registers = [self.a_reg, self.b_reg, self.res_reg]

    @property
    def name(self):
        return f"{self.add_or_mul}_{self.idx}"

    def make(self):
        return make_mul_or_add(
            self.precision,
            self.idx,
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


def make_always_tree(left, rights, comb_or_seq, fsm_idx_width):
    always_a = dedent(
        f"""\
        always @ ({'*' if comb_or_seq == 'seq' else 'posedge clock'}) begin
        """
    )
    for i, (stage, inp_a) in enumerate(rights):
        # if len(rights) == 1:
        #     cond = "begin"
        if i == 0:
            cond = f"if ({make_fsm_states([stage], fsm_idx_width)}) begin"
        else:
            cond = f"else if ({make_fsm_states([stage], fsm_idx_width)}) begin"
        # elif i < len(rights) - 1:
        # else:
        #     cond = f"else begin"

        always_a += indent(
            dedent(
                f"""\
                    {cond} // num states: {len([stage])}
                        {left} = {inp_a};
                    end
                """
            ),
            "\t",
        )
    always_a += "end"
    return always_a


class Module:
    def __init__(self, fsm_stages, precision=16):
        self.precision = precision
        self.val_graph = nx.MultiDiGraph()
        self.mul_instances: Dict[int, FMul] = {}
        self.add_instances: Dict[int, FAdd] = {}
        self.name_to_val = {}
        self.register_ids = set()
        self.wire_ids = set()

        self.max_stage_len = max(len(fs) for fs in fsm_stages)
        self._fsm_stages = dict(enumerate(fsm_stages))
        self._stage_latencies = max_latency_per_fsm_stage(self._fsm_stages)

        for idx in range(self.max_stage_len):
            mul = FMul(idx, precision)
            self.mul_instances[idx] = mul
            self.add_vals(mul.registers)
            self.add_vals([mul.res_wire])
            add = FAdd(idx, precision)
            self.add_instances[idx] = add
            self.add_vals(add.registers)
            self.add_vals([add.res_wire])

        self._max_fsm_stage = 0
        self._fsm_idx_width = 0

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

    @property
    def fsm_stages(self):
        next_fsm_stage = 0
        for i, stage in self._fsm_stages.items():
            yield next_fsm_stage, stage
            next_fsm_stage += self._stage_latencies[i] + 1

        self.max_fsm_stage = next_fsm_stage

    def add_vals(self, vals: List[Val]):
        self.val_graph.add_nodes_from(vals)
        for r in vals:
            self.name_to_val[r.name] = r
            if r.reg_or_wire == RegOrWire.REG:
                self.register_ids.add(r.id)
            if r.reg_or_wire == RegOrWire.WIRE:
                self.wire_ids.add(r.id)

    def make_assign_wire_to_reg(self):
        assigns = []
        for wire_reg in self.wire_ids.intersection(self.register_ids):
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

    def add_add_mul_transfer(
        self,
        stage,
        op_idx,
        op_type,
        inp_a: Val = None,
        inp_b: Val = None,
        res: Val = None,
        res_latency: int = None,
    ):
        op = (
            self.mul_instances[op_idx]
            if op_type == Op.MUL
            else self.add_instances[op_idx]
        )

        if inp_a is not None:
            assert inp_a in self.val_graph.nodes, inp_a
            self.val_graph.add_edge(inp_a, op.a_reg, stage=stage)

        if inp_b is not None:
            assert inp_b in self.val_graph.nodes, inp_b
            self.val_graph.add_edge(inp_b, op.b_reg, stage=stage)

        if res is not None:
            assert res in self.val_graph.nodes
            assert res_latency is not None
            self.val_graph.add_edge(op.res_reg, res, stage=stage + res_latency)


    def make_top_module_decl(self, inputs: List[Val], outputs: List[Val]):
        base_inputs = ["clock", "reset", "clock_enable"]
        input_ports = [f"[{self.precision - 1}:0] {i.name}" for i in inputs]

        base_outputs = []
        output_ports = [f"[{self.precision - 1}:0] {o.name}" for o in outputs]
        mod = "`default_nettype none\n"
        mod += "module forward (\n"
        mod += ",\n".join([f"\tinput wire {inp}" for inp in base_inputs + input_ports])
        mod += ",\n"
        mod += ",\n".join(
            [f"\toutput wire {outp}" for outp in base_outputs + output_ports]
        )
        mod += "\n);\n"

        return mod

    def make_fsm_params(self):
        params = "\n".join(
            [
                f"parameter    fsm_state{str(i).zfill(self.fsm_idx_width)} = {self.max_fsm_stage + 1}'d{1 << i};"
                for i in range(1, self.max_fsm_stage + 1)
            ]
        )
        params += "\n\n"
        params += f'(* fsm_encoding = "none" *) reg   [{self.max_fsm_stage}:0] current_state_fsm;\n'
        params += f"reg   [{self.max_fsm_stage}:0] next_state_fsm;"

        return params

    def make_fsm_wires(self):
        wires = "\n".join(
            [
                f"wire    current_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.max_fsm_stage + 1)
            ]
            + [
                f"wire    next_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.max_fsm_stage)
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

        for i in range(1, self.max_fsm_stage):
            fsm += dedent(
                f"""\
        assign next_state_fsm_state{str(i).zfill(self.fsm_idx_width)} = next_state_fsm[{fsm_2bit_width}'d{i - 1}];
            """
            )

        return fsm

    def make_trees(self):
        trees = []
        for val in self.val_graph.nodes:
            edges = self.val_graph.reverse().adj[val]
            if edges:
                arms = sorted(
                    [
                        (edge_attr["stage"], v.name)
                        for v, edge_attrs in edges.items()
                        for _, edge_attr in edge_attrs.items()
                    ]
                )
                tree = make_always_tree(
                    val.name,
                    arms,
                    comb_or_seq="seq",
                    fsm_idx_width=self.fsm_idx_width,
                )
                trees.append(tree)
        return "\n".join(trees)

    def make_add_or_mul_instances(self):
        ops = []
        for mul in self.mul_instances.values():
            ops.append(mul.make())
        for add in self.add_instances.values():
            ops.append(add.make())

        return "\n".join(ops)

    def make_registers(self):
        res = []
        for reg in self.val_graph:
            if reg.reg_or_wire == RegOrWire.WIRE:
                continue
            res.append(
                f"(* max_fanout = 50 *) reg   [{self.precision - 1}:0] {reg.name};"
            )
        return "\n".join(res)

    def make_outputs(self, outputs, program_graph):
        rets = []
        for output in outputs:
            last_reg = list(program_graph.reverse()[output.id].keys())[0]
            assert last_reg in self.register_ids, last_reg
            rets.append(f"assign {output} = {last_reg}_reg;")
        return "\n".join(rets)


def collect_op_design(mod, op_idx, op_edges):
    stage_inputs, stage_outputs, constants = {}, {}, {}
    for i, stage in mod.fsm_stages:
        if op_idx < len(stage) and stage[op_idx] in op_edges:
            op = stage[op_idx]
            edges = op_edges[op]
            stage_inputs[i] = [
                namedtuple("inp", ["u", "pos"])(u, pos) for u, _v, pos in edges
            ]
            output = [v for _u, v, _pos in edges]
            assert len(set(output)) == 1
            stage_outputs[i] = output[0]
        else:
            stage_inputs[i] = None
            stage_outputs[i] = None

    return stage_inputs, stage_outputs


def split_mul_add(stage_inputs, stage_outputs):
    input_a, input_b, input_c = {}, {}, {}
    mul_outputs = {}
    for stage, inputs in stage_inputs.items():
        if inputs is not None:
            output = stage_outputs[stage]
            mul_outputs[stage] = output

            inputs_in_position = {inp.pos: inp.u for inp in inputs}
            input_a[stage] = inputs_in_position[0]
            input_b[stage] = inputs_in_position[1]
            input_c[stage] = inputs_in_position[2]

    mul_inputs_a = input_a
    mul_inputs_b = input_b
    add_inputs_b = input_c
    return mul_inputs_a, mul_inputs_b, mul_outputs, add_inputs_b


def build_module(mod, program_graph):
    fmuladd_edges = defaultdict(list)
    for (u, v, _key), attr in program_graph.edges.items():
        if "fmuladd" not in attr["op"]:
            continue
        fmuladd_edges[attr["op"]].append((u, v, attr["pos"]))
    fadd_edges = defaultdict(list)
    for (u, v, _key), attr in program_graph.edges.items():
        if "fadd" not in attr["op"]:
            continue
        fadd_edges[attr["op"]].append((u, v, attr["pos"]))

    for op_idx in range(mod.max_stage_len):

        stage_inputs, stage_outputs = collect_op_design(mod, op_idx, fmuladd_edges)
        mul_inputs_a, mul_inputs_b, _mul_outputs, add_inputs_bs = split_mul_add(
            stage_inputs, stage_outputs
        )
        mul_res_reg = mod.mul_instances[op_idx].res_reg
        for (stage, a), (stage, b), (stage, add_b) in zip(
            mul_inputs_a.items(), mul_inputs_b.items(), add_inputs_bs.items()
        ):
            a = Val(RegOrWire.REG, a)
            b = Val(RegOrWire.REG, b)
            add_b = Val(RegOrWire.REG, add_b)
            mod.add_add_mul_transfer(stage, op_idx, Op.MUL, a, b)
            mod.add_add_mul_transfer(
                stage + MUL_LATENCY,
                op_idx,
                op_type=Op.ADD,
                inp_a=mul_res_reg,
                inp_b=add_b,
                res=add_b,
                res_latency=1,
            )

        stage_inputs, stage_outputs = collect_op_design(mod, op_idx, fadd_edges)
        for stage, stage_inputs in stage_inputs.items():
            if stage_inputs is None:
                continue
            a, b = [inp.u for inp in stage_inputs]
            a = Val(RegOrWire.REG, a)
            b = Val(RegOrWire.REG, b)
            res = Val(RegOrWire.REG, stage_outputs[stage])
            mod.add_add_mul_transfer(
                stage, op_idx, Op.ADD, a, b, res=res, res_latency=1
            )


def main(design):
    program_graph = nx.json_graph.node_link_graph(design["G"])
    fsm_stages = design["topo_sort"]
    inputs = [
        n for n, attrdict in program_graph.nodes.items() if attrdict["op"] == "input"
    ]
    outputs = [
        n for n, attrdict in program_graph.nodes.items() if attrdict["op"] == "output"
    ]
    constants = [
        n for n, attrdict in program_graph.nodes.items() if attrdict["op"] == "constant"
    ]

    mod = Module(fsm_stages, precision=16)
    mod.add_vals([Val(RegOrWire.REG, r) for r in program_graph.nodes])
    input_wires = [Val(RegOrWire.WIRE, v) for v in inputs]
    input_regs = [Val(RegOrWire.REG, v) for v in inputs]
    output_wires = [Val(RegOrWire.WIRE, v) for v in outputs]
    output_regs = [Val(RegOrWire.REG, v) for v in outputs]
    mod.add_vals(input_wires + input_regs + output_wires)
    constants = [
        Val(RegOrWire.WIRE, v) for v in design["topo_sort"][0] if "constant" in v
    ]
    mod.add_vals(constants)
    mod.add_vals([Val(RegOrWire.REG, v.id) for v in constants])

    build_module(mod, program_graph)

    print(mod.make_top_module_decl(input_wires, output_wires))
    print()
    print(mod.make_fsm_params())
    print()
    print(mod.make_fsm_wires())
    print()
    print(mod.make_registers())
    print()
    print(mod.make_add_or_mul_instances())
    print()
    print(mod.make_assign_wire_to_reg())
    print()
    print(mod.make_trees())
    print()
    print(mod.make_fsm())
    print()
    print(mod.make_outputs(output_wires, program_graph))
    print()
    print("endmodule")


if __name__ == "__main__":
    fp = sys.argv[1]
    design = json.load(open(fp))
    main(design)
