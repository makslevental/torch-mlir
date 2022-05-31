import enum
import json
import math
import struct
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
MUL_LATENCY = 2
ADD_LATENCY = 2


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
            self.precision, self.idx, self.name, self.a_reg, self.b_reg, self.res_wire, self.add_or_mul
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
    def __init__(self, max_stage_len, precision=16):
        self.max_stage_len = max_stage_len
        self.precision = precision
        self.val_graph = nx.MultiDiGraph()
        self.mul_instances: Dict[int, FMul] = {}
        self.add_instances: Dict[int, FAdd] = {}
        self.name_to_val = {}
        self.register_ids = set()
        self.wire_ids = set()

        for idx in range(max_stage_len):
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
    def max_fsm_stage(self):
        return self._max_fsm_stage

    @property
    def fsm_idx_width(self):
        return self._fsm_idx_width

    @max_fsm_stage.setter
    def max_fsm_stage(self, value):
        self._max_fsm_stage = value
        self._fsm_idx_width = math.ceil(math.log10(self.max_fsm_stage))

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
            assigns.append(dedent(f"""\
            always @ (posedge clock) begin
                {wire_reg}_reg <= {wire_reg}_wire;
            end
            """))

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
            self.max_fsm_stage = max(self.max_fsm_stage, stage + res_latency)

        self.max_fsm_stage = max(self.max_fsm_stage, stage)

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

    def make_outputs(self, outputs, ssa_graph):
        rets = []
        for output in outputs:
            last_reg = list(ssa_graph.reverse()[output.id].keys())[0]
            assert last_reg in self.register_ids, last_reg
            rets.append(f"assign {output} = {last_reg}_reg;")
        return "\n".join(rets)


def collect_op_design(op, op_idx, stages, G):
    op_subG = G.subgraph([n for n, attrdict in G.nodes.items() if attrdict["op"] == op])
    stage_inputs, stage_outputs, constants = {}, {}, {}
    for i, stage in enumerate(stages, start=1):
        if op_idx < len(stage) and stage[op_idx] in op_subG:
            val = stage[op_idx]
            preds = list(G.predecessors(val))
            stage_inputs[i] = preds
            stage_outputs[i] = val
            constants[i] = [p for p in preds if "constant" in p]

        else:
            stage_inputs[i] = None
            stage_outputs[i] = None
            constants[i] = None

    return stage_inputs, stage_outputs, constants


def split_mul_add(stage_inputs, stage_outputs, ssa_graph, var_val_to_arr_idx):
    input_a, input_b, input_c = {}, {}, {}
    outputs = {}
    for stage, inputs in stage_inputs.items():
        if inputs is not None:
            output = stage_outputs[stage]
            positions = dict(ssa_graph.reverse().adj[output])

            if output in var_val_to_arr_idx:
                output = make_var_val_reg(var_val_to_arr_idx[output])
            outputs[stage] = output

            inputs_in_position = {positions[inp]["pos"]: inp for inp in inputs}
            input_a[stage] = inputs_in_position[0]
            input_b[stage] = inputs_in_position[1]
            inp_c = inputs_in_position[2]
            if inp_c in var_val_to_arr_idx:
                inp_c = make_var_val_reg(var_val_to_arr_idx[inp_c])
            input_c[stage] = inp_c

    mul_inputs_a = input_a
    mul_inputs_b = input_b

    add_inputs_b = []
    for stage, inp in input_c.items():
        if "constant" in inp:
            add_inputs_b.append([(stage, inp)])
        else:
            add_inputs_b[-1].append((stage, inp))

    return mul_inputs_a, mul_inputs_b, add_inputs_b


def build_module(mod, ssa_graph, var_val_to_arr_idx):
    for op_idx in range(mod.max_stage_len):
        stage_inputs, stage_outputs, _constants = collect_op_design(
            "fmuladd", op_idx, design["topo_sort"][1:], ssa_graph
        )
        mul_inputs_a, mul_inputs_b, add_inputs_bs = split_mul_add(
            stage_inputs, stage_outputs, ssa_graph, var_val_to_arr_idx
        )
        for (stage, a), (stage, b) in zip(mul_inputs_a.items(), mul_inputs_b.items()):
            a = Val(RegOrWire.REG, a)
            b = Val(RegOrWire.REG, b)
            mod.add_add_mul_transfer(stage, op_idx, Op.MUL, a, b)

        mul_res_reg = mod.mul_instances[op_idx].res_reg
        for add_inputs_b in add_inputs_bs:
            first_stage, first_input = add_inputs_b[0]
            assert "constant" in first_input
            first_input = Val(RegOrWire.REG, first_input)
            mod.add_add_mul_transfer(
                first_stage, op_idx, op_type=Op.ADD, inp_b=first_input
            )
            for stage, b in add_inputs_b[1:]:
                b = Val(RegOrWire.REG, b)
                mod.add_add_mul_transfer(
                    stage + MUL_LATENCY,
                    op_idx,
                    op_type=Op.ADD,
                    inp_a=mul_res_reg,
                    inp_b=b,
                )
    for op_idx in range(mod.max_stage_len):
        stage_inputs, stage_outputs, _constants = collect_op_design(
            "fadd", op_idx, design["topo_sort"][1:], ssa_graph
        )
        for stage, stage_inputs in stage_inputs.items():
            if stage_inputs is None:
                continue
            a, b = stage_inputs
            a = Val(RegOrWire.REG, a)
            b = Val(RegOrWire.REG, b)
            if stage_outputs[stage] not in var_val_to_arr_idx:
                res = Val(RegOrWire.REG, stage_outputs[stage])
                mod.add_add_mul_transfer(
                    stage, op_idx, Op.ADD, a, b, res=res, res_latency=ADD_LATENCY
                )
            else:
                mod.add_add_mul_transfer(stage, op_idx, Op.ADD, a, b)


def main(design):
    ssa_graph = nx.jit_graph(design["G"], create_using=nx.DiGraph())
    fsm_stages = design["topo_sort"][1:]
    var_val_to_arr_idx = design["var_val_to_arr_idx"]
    # remove the last one, because it needs to produce a result
    to_delete = set()
    for i, (var, idx) in enumerate(var_val_to_arr_idx.items()):
        if i == len(var_val_to_arr_idx) - 1:
            to_delete.add(var)
            continue

        next_var, next_idx = list(var_val_to_arr_idx.items())[i + 1]
        if next_idx != idx:
            to_delete.add(var)
    for var in to_delete:
        del var_val_to_arr_idx[var]

    max_stage_len = max([len(s) for s in fsm_stages])
    registers = [v for v in ssa_graph.nodes if "var_val" in v] + list(
        set([make_var_val_reg(idx) for idx in var_val_to_arr_idx.values()])
    )
    inputs = [n for n, attrdict in ssa_graph.nodes.items() if attrdict["op"] == "input"]
    outputs = [
        n for n, attrdict in ssa_graph.nodes.items() if attrdict["op"] == "output"
    ]

    mod = Module(max_stage_len, precision=16)
    mod.add_vals([Val(RegOrWire.REG, r) for r in registers])
    input_wires = [Val(RegOrWire.WIRE, v) for v in inputs]
    input_regs = [Val(RegOrWire.REG, v) for v in inputs]
    output_wires = [Val(RegOrWire.WIRE, v) for v in outputs]
    # output_regs = [Val(RegOrWire.REG, v) for v in outputs]
    mod.add_vals(input_wires + input_regs + output_wires)
    constants = [
        Val(RegOrWire.WIRE, v) for v in design["topo_sort"][0] if "constant" in v
    ]
    mod.add_vals(constants)
    mod.add_vals([Val(RegOrWire.REG, v.id) for v in constants])

    build_module(mod, ssa_graph, var_val_to_arr_idx)

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
    print(mod.make_outputs(output_wires, ssa_graph))
    print()
    print("endmodule")


if __name__ == "__main__":
    fp = "/Users/mlevental/dev_projects/torch-mlir/hls/examples/Linear.1/design.json"
    design = json.load(open(fp))
    main(design)
