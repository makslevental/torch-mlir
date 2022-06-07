from __future__ import annotations

import itertools
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from textwrap import dedent, indent
from typing import Any, Tuple, Union

import hls.python.interpreter.mlir_ops
from hls.python.interpreter.util import get_default_args, index_map
from hls.python.interpreter.mlir_ops import ArrayIndex

PES = hls.python.interpreter.mlir_ops.PES

import numpy as np

FILE = open("forward.v", "w")

PEIndex = Tuple[int]




CURRENT_PE: PEIndex = None


class PE:
    def __init__(self, index):
        self.index = index
        self._instructions = []

    def push_instructions(self, inst):
        self._instructions.append(inst)

    def num_instructions(self):
        return len(self._instructions)

    def push_nop(self):
        self._instructions.append(NOP())


PES: Dict[PEIndex, PE] = {}

def get_val_from_global(global_name, val_idx):
    return int(str(hash((global_name, val_idx)))[:4])


def make_fsm_states(fsm_states, fsm_idx_width):
    return " | ".join(
        [
            f"(1'b1 == current_state_fsm_state{str(i).zfill(fsm_idx_width)})"
            for i in fsm_states
        ]
    )


class Module:
    def __init__(self, input_arr, output_arr, processing_elts, max_range, precision=16):
        self.input_arr = input_arr
        self.output_arr = output_arr
        self.pes = processing_elts
        self.precision = precision

        self.pe_geom_idxs = sorted(np.ndindex(*max_range))
        self.pe_geom_idx_to_idx = {idx: i for i, idx in enumerate(self.pe_geom_idxs)}
        self.pes_instructions = defaultdict(list)
        self.needed_mult_units = set()
        self.needed_add_units = set()
        self.needed_relu_units = set()
        self.fsm_steps = 0

        for pe_idx, pe in self.pes.items():
            self.fsm_steps = max(self.fsm_steps, len(pe._instructions))

            pe_geom_idx = self.pe_geom_idxs[pe_idx]
            for i, inst in enumerate(pe._instructions):
                self.pes_instructions[pe_geom_idx].append((i, inst))
                if isinstance(inst, MulInst):
                    self.needed_mult_units.add(pe_geom_idx)
                elif isinstance(inst, AddInst):
                    self.needed_add_units.add(pe_geom_idx)
                elif isinstance(inst, ReLUInst):
                    self.needed_relu_units.add(pe_geom_idx)

        self.fsm_idx_width = math.ceil(math.log10(self.fsm_steps))

    def make_top_module_decl(self):
        base_inputs = ["clock", "reset", "clock_enable"]
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
        mod = "`default_nettype none\n"
        mod += "module forward (\n"
        mod += ",\n".join([f"\tinput wire {inp}" for inp in base_inputs])
        mod += ",\n"
        mod += ",\n".join(
            [f"\toutput wire {outp}" for outp in base_outputs + output_ports]
        )
        mod += "\n);\n"

        mod += ";\n".join([f"wire {inp}" for inp in input_ports])
        mod += ";\n"

        return mod

    def make_add_instances(self):
        adds = []
        for geom_add_idx in sorted(self.needed_add_units):
            add_idx = self.pe_geom_idx_to_idx[geom_add_idx]
            id = "_".join(map(str, geom_add_idx))
            add_s = dedent(
                f"""\
                reg   [{self.precision - 1}:0] fadd_{id}_id_{add_idx}_a;
                reg   [{self.precision - 1}:0] fadd_{id}_id_{add_idx}_b;
                wire   [{self.precision - 1}:0] fadd_{id}_id_{add_idx}_z;
                reg   [{self.precision - 1}:0] fadd_{id}_id_{add_idx}_z_reg;
                
                always @ (posedge clock) begin
                    fadd_{id}_id_{add_idx}_z_reg <= fadd_{id}_id_{add_idx}_z;
                end
                
                fadd #({add_idx}) fadd_{id}_id_{add_idx}(
                    .clock(clock),
                    .reset(reset),
                    .clock_enable(clock_enable),
                    .a(fadd_{id}_id_{add_idx}_a),
                    .b(fadd_{id}_id_{add_idx}_b),
                    .z(fadd_{id}_id_{add_idx}_z)
                );
                """
            )
            adds.append(add_s)
        return "\n".join(adds)

    def make_relu_instances(self):
        relus = []
        for geom_relu_idx in sorted(self.needed_relu_units):
            relu_idx = self.pe_geom_idx_to_idx[geom_relu_idx]
            id = "_".join(map(str, geom_relu_idx))
            relu_s = dedent(
                f"""\
                reg   [{self.precision - 1}:0] frelu_{id}_id_{relu_idx}_a;
                wire   [{self.precision - 1}:0] frelu_{id}_id_{relu_idx}_z;
                relu #({relu_idx}) relu_{id}_id_{relu_idx}(
                    .a(frelu_{id}_id_{relu_idx}_a),
                    .z(frelu_{id}_id_{relu_idx}_z)
                );
                """
            )
            relus.append(relu_s)
        return "\n".join(relus)

    def make_mul_instances(self):
        muls = []
        for geom_mul_idx in sorted(self.needed_mult_units):
            mul_idx = self.pe_geom_idx_to_idx[geom_mul_idx]
            id = "_".join(map(str, geom_mul_idx))
            mul_s = dedent(
                f"""\
                reg   [{self.precision - 1}:0] fmul_{id}_id_{mul_idx}_a;
                reg   [{self.precision - 1}:0] fmul_{id}_id_{mul_idx}_b;
                wire   [{self.precision - 1}:0] fmul_{id}_id_{mul_idx}_z;
                reg   [{self.precision - 1}:0] fmul_{id}_id_{mul_idx}_z_reg;
                
                always @ (posedge clock) begin
                    fmul_{id}_id_{mul_idx}_z_reg <= fmul_{id}_id_{mul_idx}_z;
                end
                
                fmul #({mul_idx}) fmul_{id}_id_{mul_idx}(
                    .clock(clock),
                    .reset(reset),
                    .clock_enable(clock_enable),
                    .a(fmul_{id}_id_{mul_idx}_a),
                    .b(fmul_{id}_id_{mul_idx}_b),
                    .z(fmul_{id}_id_{mul_idx}_z)
                );
                """
            )
            muls.append(mul_s)
        return "\n".join(muls)

    def make_bram_instances(self):
        brams = []
        for geom_mul_idx in sorted(self.needed_mult_units):
            mul_idx = self.pe_geom_idx_to_idx[geom_mul_idx]
            id = "_".join(map(str, geom_mul_idx))
            bram_s = dedent(
                f"""\
                
                reg   [{self.precision - 1}:0] bram_{id}_id_{mul_idx}_data_reg;
                wire   [{self.precision - 1}:0] bram_{id}_id_{mul_idx}_data;
                bram #({mul_idx}, {self.precision}) bram_{id}_id_{mul_idx}(
                    .clock(clock),
                    .data_out(bram_{id}_id_{mul_idx}_data)
                );
                always @ (posedge clock) begin
                    bram_{id}_id_{mul_idx}_data_reg <= bram_{id}_id_{mul_idx}_data;
                end
                """
            )
            brams.append(bram_s)
        return "\n".join(brams)

    def make_fsm_params(self):
        params = "\n".join(
            [
                f"parameter    fsm_state{str(i).zfill(self.fsm_idx_width)} = {self.fsm_steps + 1}'d{1 << i};"
                for i in range(1, self.fsm_steps + 1)
            ]
        )
        params += "\n\n"
        params += f'(* fsm_encoding = "none" *) reg   [{self.fsm_steps}:0] current_state_fsm;\n'
        params += f"reg   [{self.fsm_steps}:0] next_state_fsm;"

        return params

    def make_fsm_wires(self):
        return "\n".join(
            [
                f"wire    current_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.fsm_steps)
            ]
            + [
                f"wire    next_state_fsm_state{str(i).zfill(self.fsm_idx_width)};"
                for i in range(0, self.fsm_steps)
            ]
        )

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
        for i in range(1, self.fsm_steps):
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
            fsm_state{str(self.fsm_steps).zfill(self.fsm_idx_width)} : begin
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

        fsm_2bit_width = math.ceil(math.log2(self.fsm_steps))
        for i in range(1, self.fsm_steps):
            fsm += dedent(
                f"""\
        assign current_state_fsm_state{str(i).zfill(self.fsm_idx_width)} = current_state_fsm[{fsm_2bit_width}'d{i - 1}];
            """
            )

        for i in range(1, self.fsm_steps):
            fsm += dedent(
                f"""\
        assign next_state_fsm_state{str(i).zfill(self.fsm_idx_width)} = next_state_fsm[{fsm_2bit_width}'d{i - 1}];
            """
            )

        return fsm

    def make_mul_left_input_case(self, pe_geom_idx):
        mul_lefts = [
            (fsm_state, inst.left)
            for fsm_state, inst in self.pes_instructions[pe_geom_idx]
            if isinstance(inst, MulInst)
        ]
        if mul_lefts:
            mul_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(mul_lefts)
            return self.make_inst_input_case(
                "mul", mul_inps_to_fsm_list, pe_geom_idx, "a"
            )
        else:
            return ""

    def make_mul_right_input_case(self, pe_geom_idx):
        mul_rights = [
            (fsm_state, inst.right)
            for fsm_state, inst in self.pes_instructions[pe_geom_idx]
            if isinstance(inst, MulInst)
        ]
        if mul_rights:
            mul_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(mul_rights)
            other_vals = [
                tup for tup in mul_inps_to_fsm_list if "__constant" not in tup[0][0]
            ]
            global_arr_vals = [
                tup for tup in mul_inps_to_fsm_list if "__constant" in tup[0][0]
            ]
            global_arr_vals = reduce(
                lambda accum, val: [
                    (("__constant", pe_geom_idx), accum[0][1] + val[1])
                ],
                global_arr_vals,
                [(("__constant", pe_geom_idx), [])],
            )
            return self.make_inst_input_case(
                "mul", other_vals + global_arr_vals, pe_geom_idx, "b"
            )
        else:
            return ""

    def make_add_left_input_case(self, pe_geom_idx):
        add_lefts = [
            (fsm_state, inst.left)
            for fsm_state, inst in self.pes_instructions[pe_geom_idx]
            if isinstance(inst, AddInst)
        ]
        if add_lefts:
            add_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(add_lefts)
            return self.make_inst_input_case(
                "add", add_inps_to_fsm_list, pe_geom_idx, "a"
            )
        else:
            return ""

    def make_add_right_input_case(self, pe_geom_idx):
        add_rights = [
            (fsm_state, inst.right)
            for fsm_state, inst in self.pes_instructions[pe_geom_idx]
            if isinstance(inst, AddInst)
        ]
        if add_rights:
            add_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(add_rights)
            return self.make_inst_input_case(
                "add", add_inps_to_fsm_list, pe_geom_idx, "b"
            )
        else:
            return ""

    def make_relu_input_case(self, pe_geom_idx):
        relu_rights = [
            (fsm_state, inst.val)
            for fsm_state, inst in self.pes_instructions[pe_geom_idx]
            if isinstance(inst, ReLUInst)
        ]
        if relu_rights:
            relu_inps_to_fsm_list = self.make_binop_inp_to_fsm_list(relu_rights)
            return self.make_inst_input_case(
                "relu", relu_inps_to_fsm_list, pe_geom_idx, "a"
            )
        else:
            return ""

    def make_binop_inp_to_fsm_list(self, insts):
        inst_inps = defaultdict(list)
        for i, inst_inp in insts:
            if isinstance(inst_inp, GlobalArrayVal):
                inst_inps[inst_inp.array.global_name, inst_inp.val_id].append(i)
            elif isinstance(inst_inp, ArrayVal) and "_arg" in inst_inp.name:
                inst_inps[inst_inp.name, inst_inp.val_id].append(i)
            elif isinstance(inst_inp, Constant):
                inst_inps["Constant", inst_inp.val_id].append(i)
            elif isinstance(inst_inp, VerilogVal) and isinstance(
                inst_inp.val_id, (AddInst, SubInst, MulInst, DivInst, ReLUInst, ExpInst)
            ):
                inst_inps[
                    inst_inp.val_id.__class__.__name__,
                    self.pe_geom_idxs[inst_inp.val_id.pe_id],
                ].append(i)
            else:
                raise Exception(f"wtfbbq {inst_inp}")

        # TODO: not the greatest thing - sort here so that most frequent case arm ends up last in if
        return list(sorted(inst_inps.items(), key=lambda x: len(x[1])))

    def make_inst_input_case(self, inst: str, inst_inps, pe_geom_idx, inp_letter):
        this_pe_geom_id = "_".join(map(str, pe_geom_idx))
        this_pe_id = self.pe_geom_idx_to_idx[pe_geom_idx]

        always = dedent(
            f"""\
            always @ (*) begin
            """
        )
        for i, ((val_name, val_id), fsm_states) in enumerate(inst_inps):
            if len(inst_inps) == 1:
                cond = "begin"
            elif i == 0:
                cond = f"if ({make_fsm_states(fsm_states, self.fsm_idx_width)}) begin"
            elif i < len(inst_inps) - 1:
                cond = (
                    f"else if ({make_fsm_states(fsm_states, self.fsm_idx_width)}) begin"
                )
            else:
                cond = f"else begin"

            _always = indent(
                dedent(
                    f"""\
                        {cond} // num states: {len(fsm_states)}
                            {{body}}
                        end 
                    """
                ),
                "\t",
            )

            if "_arg" in val_name:
                arg_name, arg_id = val_name, val_id
                arg_id = "_".join(map(str, arg_id))
                _always = _always.format(
                    body=f"f{inst}_{this_pe_geom_id}_id_{this_pe_id}_{inp_letter} = {arg_name}_{arg_id};"
                )
            elif val_name in ["AddInst", "MulInst"]:
                other_pe_geom_id = val_id
                other_pe_id = self.pe_geom_idx_to_idx[other_pe_geom_id]
                other_pe_geom_id = "_".join(map(str, other_pe_geom_id))

                if this_pe_geom_id == other_pe_geom_id:
                    res = f"f{val_name.lower().replace('inst', '')}_{other_pe_geom_id}_id_{other_pe_id}_z_reg"
                else:
                    res = f"f{val_name.lower().replace('inst', '')}_{other_pe_geom_id}_id_{other_pe_id}_z"

                _always = _always.format(
                    body=f"f{inst}_{this_pe_geom_id}_id_{this_pe_id}_{inp_letter} = {res};"
                )

            elif "ReLUInst" in val_name:
                other_pe_geom_id = val_id
                other_pe_id = self.pe_geom_idx_to_idx[other_pe_geom_id]
                other_pe_geom_id = "_".join(map(str, other_pe_geom_id))
                res = f"f{val_name.lower().replace('inst', '')}_{other_pe_geom_id}_id_{other_pe_id}_z"
                _always = _always.format(
                    body=f"f{inst}_{this_pe_geom_id}_id_{this_pe_id}_{inp_letter} = {res};"
                )

            elif "__constant" in val_name:
                assert (
                    inst == "mul"
                ), "only muls should be getting global array csts, ie from bram"
                # cst = get_val_from_global(val_name, val_id)
                _always = _always.format(
                    body=f"f{inst}_{this_pe_geom_id}_id_{this_pe_id}_{inp_letter} = bram_{this_pe_geom_id}_id_{this_pe_id}_data;"
                )
            elif "Constant" in val_name:
                cst = float(val_id)
                _always = _always.format(
                    body=f"f{inst}_{this_pe_geom_id}_id_{this_pe_id}_{inp_letter} = {self.precision}'d{half_to_int(cst)};"
                )
            else:
                warnings.warn(f"unhandled inst {val_name, val_id}")

            always += _always

        always += dedent(
            f"""\
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
            res = f"f{val.val_id.__class__.__name__.lower().replace('inst', '')}_{other_pe_geom_id}_id_{other_pe_id}_z"
            out += dedent(
                f"""\
                        assign {output.arr_name + '_' + '_'.join(map(str, idx))} = {res};
                    """
            )
        return out


def VerilogForward(input, output, forward, processing_elts, max_range):
    forward()
    module = Module(input, output, processing_elts, max_range)
    module_decl = module.make_top_module_decl()
    print(module_decl, "\n")
    fsm_params = module.make_fsm_params()
    print(fsm_params, "\n")
    fsm_wires = module.make_fsm_wires()
    print(fsm_wires, "\n")
    bram_units = module.make_bram_instances()
    print(bram_units, "\n")
    mul_units = module.make_mul_instances()
    print(mul_units, "\n")
    add_units = module.make_add_instances()
    print(add_units, "\n")
    relu_units = module.make_relu_instances()
    print(relu_units, "\n")

    for pe_geom_idx in module.pe_geom_idxs:
        always = module.make_mul_left_input_case(pe_geom_idx)
        print(always, "\n")

    for pe_geom_idx in module.pe_geom_idxs:
        always = module.make_mul_right_input_case(pe_geom_idx)
        print(always, "\n")

    for pe_geom_idx in module.pe_geom_idxs:
        always = module.make_add_left_input_case(pe_geom_idx)
        print(always, "\n")

    for pe_geom_idx in module.pe_geom_idxs:
        always = module.make_add_right_input_case(pe_geom_idx)
        print(always, "\n")

    for pe_geom_idx in module.pe_geom_idxs:
        always = module.make_relu_input_case(pe_geom_idx)
        print(always, "\n")

    outs = module.make_outputs(output)
    print(outs, "\n")

    fsm = module.make_fsm()
    print(fsm, "\n")

    print("endmodule")


def Forward(forward, max_range):
    for i, idx in enumerate(
        sorted(itertools.product(*[list(range(i)) for i in max_range]))
    ):
        hls.python.interpreter.mlir_ops.PES[i] = hls.python.interpreter.mlir_ops.PE(i)

    Args = get_default_args(forward)
    VerilogForward(
        Args["_arg0"],
        Args["_arg1"],
        forward,
        hls.python.interpreter.mlir_ops.PES,
        max_range=max_range,
    )


class VerilogVal:
    def __init__(self, name, val_id):
        self.name = name
        self.val_id = val_id

    def __repr__(self):
        return str(self.__class__.__name__)

    def __mul__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        m = MulInst(self, other)
        v = VerilogVal(f"(* {self} {other})", m)
        PES[CURRENT_PE].push_instructions(m)
        return v

    def __truediv__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        d = DivInst(self, other)
        v = VerilogVal(f"(/ {self} {other})", d)
        PES[CURRENT_PE].push_instructions(d)
        return v

    def __add__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        a = AddInst(self, other)
        v = VerilogVal(f"(+ {self} {other})", a)
        PES[CURRENT_PE].push_instructions(a)
        return v

    def __sub__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        s = SubInst(self, other)
        v = VerilogVal(f"(- {self} {other})", s)
        PES[CURRENT_PE].push_instructions(s)
        return v

    def __gt__(self, other):
        global PES
        if isinstance(other, (float, int, bool)):
            other = Constant(other)
        g = GTInst(self, other)
        v = VerilogVal(f"(> {self} {other})", g)
        PES[CURRENT_PE].push_instructions(g)
        return v


class ArrayVal(VerilogVal):
    array: ArrayDecl

    def __init__(self, name, val_id: ArrayIndex, array: ArrayDecl):
        super().__init__(name, val_id)
        self.array = array

    def __str__(self):
        return f"array{self.name}"


class GlobalArrayVal(ArrayVal):
    def __str__(self):
        return f"global{self.name}_{'_'.join(map(str, self.val_id))}"


class Constant(VerilogVal):
    def __init__(self, cst_val):
        super().__init__("cst", cst_val)


def ReLU(*args):
    def op(arg):
        pe = PES[CURRENT_PE]
        r = ReLUInst(arg)
        v = VerilogVal(f"(relu {arg})", r)
        pe.push_instructions(r)
        return v

    return op


def Exp(*args):
    def op(arg):
        pe = PES[CURRENT_PE]
        r = ExpInst(arg)
        v = VerilogVal(f"(exp {arg})", r)
        pe.push_instructions(r)
        return v

    return op


class ArrayDecl:
    def __init__(self, arr_name, *shape, input=False, output=False):
        self.arr_name = arr_name
        self.curr_shape = shape
        self.prev_shape = shape
        self.pe_index = shape
        self.registers = {}
        self.input = input
        self.output = output

    def __getitem__(self, index: ArrayIndex):
        global PES
        try:
            index = self.idx_map(index)
        except ValueError:
            index = (-1, -1, -1, -1)

        if index not in self.registers:
            if not self.input:
                v = Constant("0.0")
            else:
                v = ArrayVal(f"{self.arr_name}", index, self)
            self.registers[index] = v

        v = self.registers[index]
        return v

    def __setitem__(self, index, value):
        global PES
        try:
            index = self.idx_map(index)
        except ValueError:
            index = (-1, -1, -1, -1)
        assert not self.input
        self.registers[index] = value

    def idx_map(self, index):
        return index_map(index, self.curr_shape, self.prev_shape)

    def reshape(self, *shape):
        self.prev_shape = self.curr_shape
        self.curr_shape = shape
        return self


class GlobalArray:
    def __init__(self, local_name, global_name, global_array):
        self.name = local_name
        self.global_name = global_name
        self.global_array = global_array
        self.curr_shape = global_array.shape

    def __getitem__(self, index: ArrayIndex):
        v = GlobalArrayVal(self.name, index, self)
        return v


def ParFor(body, ranges):
    global PES, CURRENT_PE
    pes_run = set()
    num_insts = None
    for i, idx in enumerate(sorted(itertools.product(*ranges))):
        CURRENT_PE = i
        cur_num_insts = PES[CURRENT_PE].num_instructions()
        body(*idx)
        pes_run.add(i)
        if num_insts is None:
            num_insts = PES[CURRENT_PE].num_instructions() - cur_num_insts

    for pe_idx, pe in PES.items():
        if pe_idx not in pes_run:
            for _ in range(num_insts):
                pe.push_nop()


from llvmlite import binding as llvm

if __name__ == "__main__":
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llmod = llvm.parse_assembly(
        open(
            "/home/mlevental/dev_projects/torch-mlir/hls/examples/BraggNN.1/forward.opt.ll"
        ).read()
    )
    print(llmod)


@dataclass(frozen=True)
class _Instruction:
    pe_id: PEIndex = field(init=False, default_factory=lambda: CURRENT_PE)


@dataclass(frozen=True)
class NOP(_Instruction):
    pass


@dataclass(frozen=True)
class Bin(_Instruction):
    left: Any
    right: Any


@dataclass(frozen=True)
class MulInst(Bin):
    pass


@dataclass(frozen=True)
class DivInst(Bin):
    pass


@dataclass(frozen=True)
class AddInst(Bin):
    pass

@dataclass(frozen=True)
class SubInst(Bin):
    pass


@dataclass(frozen=True)
class GTInst(Bin):
    pass


@dataclass(frozen=True)
class ReLUInst(_Instruction):
    val: Any


@dataclass(frozen=True)
class ExpInst(_Instruction):
    val: Any

Instruction = Union[NOP, ExpInst, ReLUInst, MulInst, AddInst, DivInst, SubInst, GTInst]
