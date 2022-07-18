import argparse
import enum
import os
from collections import namedtuple

from dataclasses import dataclass

from hls.scripts.refactor.ops import OpType, LATENCIES
from hls.scripts.refactor.parse import parse_mlir_module
from hls.scripts.refactor.rtl.basic import Wire, Reg, make_constant, make_always_tree
from hls.scripts.refactor.rtl.fsm import FSM
from hls.scripts.refactor.rtl.ip import FAdd, FMul, ReLU, Neg, PE
from hls.scripts.refactor.rtl.module import make_top_module_decl

DEBUG = True


def make_op_always(data, fsm, pes, vals, input_wires):
    op = data.opr
    ip = getattr(pes[data.pe_idx], op.value, None)
    args = data.op_inputs
    trees = []
    start_time = data.start_time
    end_time = start_time + LATENCIES[op]
    res_val = vals.get(data.res_val, data.res_val)
    in_a = vals.get(args[0], input_wires.get(args[0], args[0]))
    if op in {OpType.MUL, OpType.DIV, OpType.ADD, OpType.SUB, OpType.GT}:
        in_b = vals.get(args[1], input_wires.get(args[1], args[1]))
        trees.append(make_always_tree(ip.x, in_a, fsm.make_fsm_states([start_time])))
        trees.append(make_always_tree(ip.y, in_b, fsm.make_fsm_states([start_time])))
        trees.append(make_always_tree(res_val, ip.r, fsm.make_fsm_states([end_time])))
    elif op in {OpType.NEG, OpType.RELU}:
        trees.append(make_always_tree(ip.a, in_a, fsm.make_fsm_states([start_time])))
        trees.append(make_always_tree(res_val, ip.res, fsm.make_fsm_states([end_time])))
    elif op in {OpType.COPY}:
        trees.append(make_always_tree(res_val, in_a, fsm.make_fsm_states([start_time])))
    else:
        raise NotImplementedError

    return "\n".join(trees)


def main(mac_rewritten_sched_mlir_fp, precision):
    mac_rewritten_sched_mlir_str = open(mac_rewritten_sched_mlir_fp).read()
    (
        op_id_data,
        func_args,
        returns,
        return_time,
        vals,
        csts,
        pe_idxs,
    ) = parse_mlir_module(mac_rewritten_sched_mlir_str)
    input_wires = {v: Wire(v, precision) for v in func_args}
    output_wires = {v: Wire(v, precision) for v in returns}
    vals = {v: Reg(v, precision) for v in vals}

    verilog_file = open(mac_rewritten_sched_mlir_fp.replace(".mlir", ".v"), "w")

    def emit(*args):
        print(*args, file=verilog_file)
        print(file=verilog_file)

    def debug_print(self, *args):
        if DEBUG:
            self.emit(["//"] + args)

    emit(
        make_top_module_decl(
            list(input_wires.values()),
            list(f"output_{v}" for v in output_wires.values()),
            precision,
        )
    )
    for name, val_reg in vals.items():
        if name in csts:
            emit(val_reg.instantiate()[:-1], "=", f"{make_constant(None, precision)};")
        else:
            emit(val_reg.instantiate())

    fsm = FSM(50, max_fsm_stage=return_time)
    emit(fsm.make_fsm_params())
    emit(fsm.make_fsm_wires())

    pes = {}
    for pe_idx in pe_idxs:
        if pe_idx[0] < 0:
            continue

        fadd = FAdd(pe_idx, precision)
        emit(fadd.instantiate())
        fmul = FMul(pe_idx, precision)
        emit(fmul.instantiate())
        relu = ReLU(pe_idx, precision)
        emit(relu.instantiate())
        neg = Neg(pe_idx, precision)
        pes[pe_idx] = PE(fadd, fmul, relu, neg)

    for (op_id, op), data in op_id_data.items():
        if op == OpType.CST:
            continue
        emit(make_op_always(data, fsm, pes, vals, input_wires))

    emit(fsm.make_fsm())

    for v, wire in output_wires.items():
        emit(f"assign output_{v} = {v};")

    emit("endmodule")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fp")
    parser.add_argument("--precision", default=11)
    args = parser.parse_args()
    main(args.fp, args.precision)
