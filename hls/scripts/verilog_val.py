from collections import defaultdict, deque
from itertools import product
from textwrap import dedent, indent

import numpy as np


def format_cst(cst):
    return np.random.randint(1, 100000)
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


REG_COUNT = 0
WIRE_COUNT = 0


def make_id(idx):
    return idx


def make_mac(idx):
    id = make_id(idx)

    return f"""
    
    wire aresetn_{id};
    reg[15:0] a_tdata_{id};
    wire a_tlast_{id};
    reg[15:0] b_tdata_{id};
    wire[15:0] r_tdata_{id};
    wire r_tlast_{id};
     
    mac mac_{id} (
        .aclk(ap_clk),
        .aresetn(aresetn_{id}),
        .a_tdata(a_tdata_{id}),
        .a_tlast(a_tlast_{id}),
        .b_tdata(b_tdata_{id}),
        .r_tdata(r_tdata_{id}),
        .r_tlast(r_tlast_{id})
    );
    """


def make_mac_output(idx):
    id = make_id(idx)
    return f"r_tdata_{id}_reg <= r_tdata_{id}"


class VerilogWire:
    def __init__(self, name):
        global WIRE_COUNT
        self.name = f"{name}"
        self.var_id = WIRE_COUNT
        WIRE_COUNT += 1

    def __str__(self):
        return f"{self.name}"

    def __mul__(self, other):
        # <result> = fmul float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = VerilogWire(other)
        v = VerilogWire(f"(* ({self}) ({other}))")
        # print(f"{v} = fmul float {self}, {other}")
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float {other}, float 0.0)"
        # )
        return VerilogWire(f"{format_cst(None)} * {format_cst(None)}")

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = VerilogWire(other)
        v = VerilogWire(f"(+ ({self}) ({other}))")
        # print(f"{v} = fadd float {self}, {other}")
        # print(
        #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float 1.0, float {other})"
        # )
        return VerilogWire(f"{format_cst(None)} + {format_cst(None)}")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = VerilogWire(other)
        v = VerilogWire(f"(> ({self}) ({other}))")
        # print(f"{v} = fcmp ugt float {self}, {other}")
        return VerilogWire(f"{format_cst(None)} > {format_cst(None)}")


class VerilogReg:
    def __init__(self, name):
        global REG_COUNT
        self.name = f"{name}"
        self.var_id = REG_COUNT
        REG_COUNT += 1

    def __str__(self):
        return f"{self.name}"


class VerilogConstant(VerilogWire):
    def __init__(self, name):
        super(VerilogConstant, self).__init__(name)
        self._fmt = f"16'd{format_cst(self.name)}"

    def __str__(self):
        return self._fmt


MAC_IDX: int = ()


class MAC_TERMINAL:
    def __init__(self, parfor_idx):
        self.parfor_idx = parfor_idx

    def __str__(self):
        return str(self.parfor_idx)


MAC_STACKS = defaultdict(deque)

MACS = {}


class MAC(VerilogWire):
    def __init__(self, a, b, c):
        super(MAC, self).__init__(str(MAC_IDX))
        self.mac_idx = MAC_IDX
        inps = [a, b, c]
        for i, v in enumerate(inps):
            if isinstance(v, (float, int, bool)):
                inps[i] = VerilogConstant(v)

        a, b, c = inps
        self.a = VerilogWire(f"a_tdata_{MAC_IDX}")
        self.b = VerilogWire(f"b_tdata_{MAC_IDX}")
        self.r_wire = VerilogWire(f"r_tdata_{MAC_IDX}")
        # TODO: start of accum
        if isinstance(b, VerilogConstant) and isinstance(c, VerilogConstant):
            self.set_mac_wires(c)
        self.set_mac_wires(a, b)
        MACS[MAC_IDX] = self

    def __call__(self, *args, **kwargs):
        return self.r_wire

    def set_mac_wires(self, a=None, b=None, r=None):
        a_wire, b_wire, r_wire = self.a, self.b, self.r_wire
        if a is not None and b is not None:
            MAC_STACKS[MAC_IDX].append(((f"{a_wire} <= {a}"), (f"{b_wire} <= {b}")))
        elif a is not None and b is None:
            MAC_STACKS[MAC_IDX].append(((f"{a_wire} <= {a}"), (f"{b_wire} <= 16'd0")))
        elif a is None and b is None:
            assert r is not None
            MAC_STACKS[MAC_IDX].append((f"{r} <= {r_wire}",))


def make_args_globals(Args):
    from mlir_ops import ArrayDecl

    args = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            if arg.input:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(
                        f"\tinput [15:0] {arg.var_name}_{'_'.join(map(str, index))}"
                    )
            elif arg.output:
                for index in np.ndindex(*arg.curr_shape):
                    args.append(
                        f"\toutput [15:0] {arg.var_name}_{'_'.join(map(str, index))}"
                    )

    return args


FILE = open("forward.v", "w")


# FILE = sys.stdout


def any_rounds_left():
    return any([len(macs) for macs in MAC_STACKS.values()])


def VerilogForward(Args, OUTPUT_ARRAYS, forward):
    args = make_args_globals(Args)

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

    for i, inp in enumerate(args):
        print(inp, end="", file=FILE)
        if i < len(args) - 1:
            print(",\n", end="", file=FILE)
        else:
            print("\n", end="", file=FILE)

    print(");", file=FILE)

    forward()

    rounds = []

    for mac_idx in MAC_STACKS:
        print(make_mac(mac_idx), file=FILE)

    for mac_idx, macs in MAC_STACKS.items():
        print(indent("always @(posedge ap_clk) begin", "\t"), file=FILE)
        for mac in macs:
            for assign in mac:
                print(indent(f'{assign.replace("<=", "=")};', "\t\t"), file=FILE)
        print(indent("end", "\t"), file=FILE)
        print(file=FILE)

    # while any_rounds_left():
    #     round = []
    #     for mac_idx, macs in MAC_STACKS.items():
    #         round.append(macs.popleft())
    #     rounds.append(round)
    #
    # for i, round in enumerate(rounds):
    #     if not len(round) or isinstance(round[0], MAC_TERMINAL):
    #         # TODO end accum
    #         continue
    #
    #     print(indent("always @(*) begin", "\t"), file=FILE)
    #     for assigns in round:
    #         for assign in assigns:
    #             print(indent(f"{assign};", "\t\t"), file=FILE)
    #     print(indent("end", "\t"), file=FILE)
    #     print(file=FILE)

    for arr in OUTPUT_ARRAYS:
        for idx, reg in arr.registers.items():
            out_wire = "_arg1_" + "_".join(map(str, idx))
            print(indent(f"assign {out_wire} = {reg};", "\t"), file=FILE)
        print(file=FILE)

    print("endmodule", file=FILE)


def test_hex():
    print(format_cst(1000 + np.random.randn(1)[0]))


PAR_IDX = 0


def ParFor(body, ranges):
    global PAR_IDX
    global MAC_IDX

    for i, idxs in enumerate(product(*ranges)):
        MAC_IDX = i
        body(*idxs)
        # if len(MAC_STACKS[MAC_IDX]):
        #     MAC_STACKS[MAC_IDX].append((make_mac_output(MAC_IDX),))
        # MAC_STACKS[MAC_IDX].append(MAC_TERMINAL(PAR_IDX))

    PAR_IDX += 1
