import math
import pickle
import struct
import warnings
from collections import defaultdict
from textwrap import dedent, indent

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


FILE = open("forward.v", "w")


def test_hex():
    print(format_cst(1000 + np.random.randn(1)[0]))


def get_val_from_global(global_name, val_idx):
    return int(str(hash((global_name, val_idx)))[:4])


def make_fsm_states(fsm_states, fsm_idx_width):
    return " | ".join(
        [f"(1'b1 == ap_CS_fsm_state{str(i).zfill(fsm_idx_width)})" for i in fsm_states]
    )


class Module:
    def __init__(self, input_arr, output_arr, processing_elts, max_range, precision=16):
        from mlir_ops import MulInst, AddInst

        self.input_arr = input_arr
        self.output_arr = output_arr
        self.pes = processing_elts
        self.precision = precision

        self.pe_geom_idxs = sorted(np.ndindex(*max_range))
        self.pe_geom_idx_to_idx = {idx: i for i, idx in enumerate(self.pe_geom_idxs)}
        self.pes_instructions = defaultdict(lambda: defaultdict(list))
        self.needed_mult_units = set()
        self.needed_add_units = set()
        self.fsm_steps = 0

        for pe_idx, pe in self.pes.items():
            self.fsm_steps = max(self.fsm_steps, len(pe._instructions))

            pe_geom_idx = self.pe_geom_idxs[pe_idx]
            for i, inst in enumerate(pe._instructions):
                self.pes_instructions[pe_geom_idx]["all"].append((i, inst))

                if isinstance(inst, MulInst):
                    self.pes_instructions[pe_geom_idx]["mul"].append((i, inst))
                    self.needed_mult_units.add(pe_geom_idx)
                if isinstance(inst, AddInst):
                    self.pes_instructions[pe_geom_idx]["add"].append((i, inst))
                    self.needed_add_units.add(pe_geom_idx)

        self.fsm_idx_width = math.ceil(math.log10(self.fsm_steps))

    def make_top_module_decl(self):
        base_inputs = ["ap_clk", "ap_rst", "ap_start"]
        idxs = sorted(np.ndindex(*self.input_arr.curr_shape))
        input_ports = [
            f"[{self.precision - 1}:0] {self.input_arr.arr_name}_{'_'.join(map(str, idx))}"
            for idx in idxs
        ]

        base_outputs = []
        idxs = sorted(np.ndindex(*self.output_arr.curr_shape))
        output_ports = [
            f"[{self.precision - 1}:0] {self.output_arr.arr_name}_{'_'.join(map(str, idx))}"
            for idx in idxs
        ]
        mod = "module forward (\n"
        mod += ",\n".join([f"\tinput {inp}" for inp in base_inputs + input_ports])
        mod += ",\n"
        mod += ",\n".join([f"\toutput {outp}" for outp in base_outputs + output_ports])
        mod += "\n);"

        return mod

    def make_add_instances(self):
        adds = []
        for geom_add_idx in self.needed_add_units:
            add_idx = self.pe_geom_idx_to_idx[geom_add_idx]
            id = "_".join(map(str, geom_add_idx))
            add_s = dedent(
                f"""\
                reg   [{self.precision - 1}:0] grp_fadd_{id}_fu_{add_idx}_a;
                reg   [{self.precision - 1}:0] grp_fadd_{id}_fu_{add_idx}_b;
                wire   [{self.precision - 1}:0] grp_fadd_{id}_fu_{add_idx}_z;
                reg   [{self.precision - 1}:0] grp_fadd_{id}_fu_{add_idx}_z_reg;
                
                always @ (posedge ap_clk) begin
                    grp_fadd_{id}_fu_{add_idx}_z_reg <= grp_fadd_{id}_fu_{add_idx}_z;
                end
                
                fadd #({add_idx}) grp_fadd_{id}_fu_{add_idx}(
                    .ap_clk(ap_clk),
                    .ap_rst(ap_rst),
                    .a(grp_fadd_{id}_fu_{add_idx}_a),
                    .b(grp_fadd_{id}_fu_{add_idx}_b),
                    .z(grp_fadd_{id}_fu_{add_idx}_z)
                );
                """
            )
            adds.append(add_s)
        return "\n".join(adds)

    def make_mul_instances(self):
        muls = []
        for geom_mul_idx in self.needed_mult_units:
            mul_idx = self.pe_geom_idx_to_idx[geom_mul_idx]
            id = "_".join(map(str, geom_mul_idx))
            mul_s = dedent(
                f"""\
                reg   [{self.precision - 1}:0] grp_fmul_{id}_fu_{mul_idx}_a;
                reg   [{self.precision - 1}:0] grp_fmul_{id}_fu_{mul_idx}_b;
                wire   [{self.precision - 1}:0] grp_fmul_{id}_fu_{mul_idx}_z;
                reg   [{self.precision - 1}:0] grp_fmul_{id}_fu_{mul_idx}_z_reg;
                
                always @ (posedge ap_clk) begin
                    grp_fmul_{id}_fu_{mul_idx}_z_reg <= grp_fmul_{id}_fu_{mul_idx}_z;
                end
                
                fmul #({mul_idx}) grp_fmul_{id}_fu_{mul_idx}(
                    .ap_clk(ap_clk),
                    .ap_rst(ap_rst),
                    .a(grp_fmul_{id}_fu_{mul_idx}_a),
                    .b(grp_fmul_{id}_fu_{mul_idx}_b),
                    .z(grp_fmul_{id}_fu_{mul_idx}_z)
                );
                """
            )
            muls.append(mul_s)
        return "\n".join(muls)

    def make_fsm_params(self):
        params = "\n".join(
            [
                f"parameter    ap_ST_fsm_state{str(i).zfill(self.fsm_idx_width)} = {self.fsm_steps + 1}'d{1 << i};"
                for i in range(1, self.fsm_steps + 1)
            ]
        )
        params += "\n\n"
        params += f'(* fsm_encoding = "none" *) reg   [{self.fsm_steps}:0] ap_CS_fsm;'
        params += f"reg   [{self.fsm_steps}:0] ap_NS_fsm;"

        return params

    def make_fsm_wires(self):
        return "\n".join(
            [
                f"wire    ap_CS_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.fsm_steps)
            ] +
            [
                f"wire    ap_NS_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.fsm_steps)
            ]
        )

    def make_fsm(self):
        first_state = str(1).zfill(self.fsm_idx_width)
        second_state = str(2).zfill(self.fsm_idx_width)
        fsm = dedent(
            f"""\
            always @ (posedge ap_clk) begin
                if (ap_rst == 1'b1) begin
                    ap_CS_fsm <= ap_ST_fsm_state{first_state};
                end else begin
                    ap_CS_fsm <= ap_NS_fsm;
                end
            end
            
            always @ (*) begin
                case (ap_CS_fsm)
            """
        )
        for i in range(1, self.fsm_steps):
            fsm += indent(
                dedent(
                    f"""\
                ap_ST_fsm_state{str(i).zfill(self.fsm_idx_width)} : begin
                    ap_NS_fsm = ap_ST_fsm_state{str(i + 1).zfill(self.fsm_idx_width)};
                end
                """
                ),
                "\t\t",
            )

        fsm += indent(
            dedent(
                f"""\
            ap_ST_fsm_state{str(self.fsm_steps).zfill(self.fsm_idx_width)} : begin
                ap_NS_fsm = ap_ST_fsm_state{str(1).zfill(self.fsm_idx_width)};
            end
            """
            ),
            "\t\t",
        )

        fsm += indent(
            dedent(
                """\
        default : begin
            ap_NS_fsm = 'bx;
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

        for i in range(1, self.fsm_steps + 1):
            fsm += dedent(
                f"""\
        assign ap_CS_fsm_state{str(i).zfill(self.fsm_idx_width)} = ap_CS_fsm[32'd{i - 1}];
            """
            )

        for i in range(1, self.fsm_steps + 1):
            fsm += dedent(
                f"""\
        assign ap_NS_fsm_state{str(i).zfill(self.fsm_idx_width)} = ap_NS_fsm[32'd{i - 1}];
            """
            )

        return fsm

    def make_mul_left_input_case(self, pe_geom_idx):
        mul_lefts = [(fsm_state, mul.left) for fsm_state, mul in self.pes_instructions[pe_geom_idx]["mul"]]
        mul_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(mul_lefts)
        return self.make_inst_input_case("mul", mul_inps_to_fsm_list, pe_geom_idx, "a")

    def make_mul_right_input_case(self, pe_geom_idx):
        mul_rights = [(fsm_state, mul.right) for fsm_state, mul in self.pes_instructions[pe_geom_idx]["mul"]]
        mul_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(mul_rights)
        return self.make_inst_input_case("mul", mul_inps_to_fsm_list, pe_geom_idx, "b")

    def make_add_left_input_case(self, pe_geom_idx):
        add_lefts = [(fsm_state, add.left) for fsm_state, add in self.pes_instructions[pe_geom_idx]["add"]]
        add_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(add_lefts)
        return self.make_inst_input_case("add", add_inps_to_fsm_list, pe_geom_idx, "a")

    def make_add_right_input_case(self, pe_geom_idx):
        add_rights = [(fsm_state, add.right) for fsm_state, add in self.pes_instructions[pe_geom_idx]["add"]]
        add_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(add_rights)
        return self.make_inst_input_case("add", add_inps_to_fsm_list, pe_geom_idx, "b")


    def make_binop_inp_to_fsm_list(self, insts):
        from mlir_ops import (
            ArrayDecl,
            GlobalArrayVal,
            ArrayVal,
            Val,
            AddInst,
            MulInst,
            DivInst,
            ReLUInst,
            Constant, ExpInst, SubInst,
        )

        inst_inps = defaultdict(list)
        for i, inst_inp in insts:
            if isinstance(inst_inp, GlobalArrayVal):
                inst_inps[inst_inp.array.global_name, inst_inp.val_id].append(i)
            elif isinstance(inst_inp, ArrayVal) and "_arg" in inst_inp.name:
                inst_inps[inst_inp.name, inst_inp.val_id].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, AddInst):
                inst_inps["Add", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, SubInst):
                inst_inps["Sub", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, MulInst):
                inst_inps["Mul", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, DivInst):
                inst_inps["Div", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, ReLUInst):
                inst_inps["ReLU", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Val) and isinstance(inst_inp.val_id, ExpInst):
                inst_inps["Exp", self.pe_geom_idxs[inst_inp.val_id.pe_id]].append(i)
            elif isinstance(inst_inp, Constant):
                inst_inps["Constant", inst_inp.val_id].append(i)
            else:
                raise Exception(f"wtfbbq {inst_inp}")

        return list(inst_inps.items())

    def make_inst_input_case(self, inst: str, inst_inps, pe_geom_idx, inp_letter):
        first_val_id, first_fsm_states = inst_inps[0]
        first_cst = get_val_from_global(*first_val_id)

        this_pe_geom_id = "_".join(map(str, pe_geom_idx))
        this_pe_id = self.pe_geom_idx_to_idx[pe_geom_idx]
        always = dedent(
            f"""\
            always @ (*) begin
                if ({make_fsm_states(first_fsm_states, self.fsm_idx_width)}) begin
                    grp_f{inst}_{this_pe_geom_id}_fu_{this_pe_id}_{inp_letter} = {self.precision}'d{half_to_int(first_cst)};
                end 
            """
        )
        for (val_name, val_id), fsm_states in inst_inps[1:]:
            if "_arg" in val_name:
                arg_name, arg_id = val_name, val_id
                arg_id = "_".join(map(str, arg_id))
                always += indent(
                    dedent(
                        f"""\
                            else if ({make_fsm_states(fsm_states, self.fsm_idx_width)}) begin
                                grp_f{inst}_{this_pe_geom_id}_fu_{this_pe_id}_{inp_letter} = {arg_name}_{arg_id};
                            end 
                        """
                    ),
                    "\t",
                )
            elif val_name in ["Add", "Mul"]:
                other_pe_geom_id = val_id
                other_pe_id = self.pe_geom_idx_to_idx[other_pe_geom_id]
                other_pe_geom_id = "_".join(map(str, other_pe_geom_id))
                if val_name == "Mul":
                    assert this_pe_geom_id != other_pe_geom_id or inst == 'mul'

                if this_pe_geom_id == other_pe_geom_id:
                    res = f"grp_f{val_name.lower()}_{other_pe_geom_id}_fu_{other_pe_id}_z_reg"
                else:
                    res = f"grp_f{val_name.lower()}_{other_pe_geom_id}_fu_{other_pe_id}_z"

                always += indent(
                    dedent(
                        f"""\
                            else if ({make_fsm_states(fsm_states, self.fsm_idx_width)}) begin
                                grp_f{inst}_{this_pe_geom_id}_fu_{this_pe_id}_{inp_letter} = {res};
                            end 
                        """
                    ),
                    "\t",
                )
            elif "__constant" in val_name:
                cst = get_val_from_global(val_name, val_id)
                always += indent(
                    dedent(
                        f"""\
                    else if ({make_fsm_states(fsm_states, self.fsm_idx_width)}) begin
                        grp_f{inst}_{this_pe_geom_id}_fu_{this_pe_id}_{inp_letter} = {self.precision}'d{half_to_int(cst)};
                    end 
                """
                    ),
                    "\t",
                )
            else:
                warnings.warn(f"unhandled inst {val_name, val_id}")

        always += dedent(
            f"""\
                else begin
                    grp_f{inst}_{this_pe_geom_id}_fu_{this_pe_id}_{inp_letter} = 'bx;
                end
            end
            """
        )

        return always

    def make_outputs(self, output):
        out = ""
        for idx, val in output.registers.items():
            other_pe_id = val.val_id.pe_id
            other_pe_geom_id = self.pe_geom_idxs[other_pe_id]
            other_pe_geom_id = "_".join(map(str, other_pe_geom_id))
            res = f"grp_f{val.val_id.__class__.__name__.lower().replace('inst', '')}_{other_pe_geom_id}_fu_{other_pe_id}_z"
            out += dedent(f"""\
                        assign {output.arr_name +'_' + '_'.join(map(str, idx))} = {res};
                    """)
        return out

def VerilogForward(input, output, forward, processing_elts, max_range):
    # forward()
    module = Module(input, output, processing_elts, max_range)
    module_decl = module.make_top_module_decl()
    # print(module_decl)
    # fsm_params = module.make_fsm_params()
    # print(fsm_params)
    # fsm_wires = module.make_fsm_wires()
    # print(fsm_wires)
    # mul_units = module.make_mul_instances()
    # print(mul_units)
    # add_units = module.make_add_instances()
    # print(add_units)
    #
    # for pe_geom_idx in module.pe_geom_idxs:
    #     always = module.make_mul_right_input_case(pe_geom_idx)
    #     print(always)
    #
    # for pe_geom_idx in module.pe_geom_idxs:
    #     always = module.make_mul_left_input_case(pe_geom_idx)
    #     print(always)
    #
    # for pe_geom_idx in module.pe_geom_idxs:
    #     always = module.make_add_left_input_case(pe_geom_idx)
    #     print(always)

    outs = module.make_outputs(output)
    print(outs)

    fsm = module.make_fsm()
    print(fsm)

    print("endmodule")


if __name__ == "__main__":
    from mlir_ops import (
        ArrayDecl,
        GlobalArrayVal,
        ArrayVal,
        Val,
        AddInst,
        MulInst,
        DivInst,
        ReLUInst,
        Constant, ExpInst, SubInst,
    )

    processing_elts = pickle.load(open("pes.pkl", "rb"))
    VerilogForward(
        ArrayDecl("_arg0", 1, 1, 11, 11, input=True),
        ArrayDecl("_arg1", 1, 2, output=True),
        None,
        processing_elts=pickle.load(open("pes.pkl", "rb")),
        max_range=(1, 16, 9, 9),
    )
