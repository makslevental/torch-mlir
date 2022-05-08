from collections import defaultdict, deque
from itertools import product
from textwrap import dedent, indent

import numpy as np


def format_cst(cst):
    return np.random.randint(1, 100000)
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


REG_COUNT = 0
WIRE_COUNT = 0


def make_mac(id):
    return f"""
    
    wire aresetn_{id};
    reg[15:0] a_tdata_{id};
    wire a_tlast_{id};
    reg[15:0] b_tdata_{id};
    wire[15:0] r_tdata_{id};
    reg[15:0] r_tdata_{id}_reg;
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
    mac = MACS[idx]
    return f"r_tdata_{mac.id}_reg <= r_tdata_{mac.id}"


def make_relu(id):
    return f"""
    
    reg[15:0] din_relu_{id};
    wire[15:0] dout_relu_{id};
    reg[15:0] dout_relu_{id}_reg;
     
    relu relu_{id} (
        .din_relu(din_relu_{id}),
        .dout_relu(dout_relu_{id})
    );
    """


def make_relu_output(idx):
    relu = RELUS[idx]
    return f"dout_relu_{relu.id}_reg <= dout_relu_{relu.id}"


class VerilogWire:
    def __init__(self, name):
        global WIRE_COUNT
        self.name = f"{name}"
        self.var_id = WIRE_COUNT
        WIRE_COUNT += 1

    def __str__(self):
        return f"{self.name}"

    # def __mul__(self, other):
    #     # <result> = fmul float 4.0, %var
    #     if isinstance(other, (float, int, bool)):
    #         other = VerilogWire(other)
    #     v = VerilogWire(f"(* ({self}) ({other}))")
    #     # print(f"{v} = fmul float {self}, {other}")
    #     # print(
    #     #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float {other}, float 0.0)"
    #     # )
    #     return VerilogWire(f"{format_cst(None)} * {format_cst(None)}")

    # def __add__(self, other):
    #     # <result> = fadd float 4.0, %var
    #     if isinstance(other, (float, int, bool)):
    #         other = VerilogWire(other)
    #     v = VerilogWire(f"(+ ({self}) ({other}))")
    #     # print(f"{v} = fadd float {self}, {other}")
    #     # print(
    #     #     f"{v} = call float @llvm.fmuladd.f32(float {self}, float 1.0, float {other})"
    #     # )
    #     return VerilogWire(f"{format_cst(None)} + {format_cst(None)}")

    # def __gt__(self, other):
    #     # <result> = fcmp ugt float 4.0, 5.0
    #     if isinstance(other, (float, int, bool)):
    #         other = VerilogWire(other)
    #     v = VerilogWire(f"(> ({self}) ({other}))")
    #     # print(f"{v} = fcmp ugt float {self}, {other}")
    #     return VerilogWire(f"{format_cst(None)} > {format_cst(None)}")


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


MAC_QUEUES = defaultdict(deque)



class MMAC:
    def __init__(self, MAC_IDX):
        self.mac_idx = MAC_IDX
        self.id = '_'.join(map(str, MAC_IDX))
        self.a_wire = VerilogWire(f"a_tdata_{self.id}")
        self.b_wire = VerilogWire(f"b_tdata_{self.id}")
        self.r_wire = VerilogWire(f"r_tdata_{self.id}")
        self.work = deque()
    
    def push_work(self, a, b):
        self.work.append(f"{self.a_wire} <= {a}")
        self.work.append(f"{self.b_wire} <= {b}")
    
    def push_read_result(self, r):
            self.work.append(f"{r} <= {self.r_wire}")


MACS = {}


def MAC(*idx):
    if idx not in MACS:
        MACS[idx] = MMAC(idx)
    mac = MACS[idx]
    
    def op(*args, **kwargs):
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = VerilogConstant(v)

        if kwargs["type"] == 'MulAdd':
            a, b, c = args
            # TODO: start of accum
            if isinstance(b, VerilogConstant) and isinstance(c, VerilogConstant):
                # TODO: BRAM move here
                mac.push_work(c, "16'b0")
            mac.push_work(a, b)
        else:
            a, b = args
            if kwargs["type"] == "Add":
                # TODO: this is actually 4 cycles
                # need wires out of the MAC
                mac.push_work(a, "16'b0")
                mac.push_work(b, "16'b0")
            elif kwargs["type"] == "Mult":
                mac.push_work("16'b0", "16'b0")
                mac.push_work(a, b)
        
        return mac.r_wire

    return op

RELUS = {}

class RReLU:
    def __init__(self, idx):
        self.idx = idx
        self.id = '_'.join(map(str, idx))
        self.in_wire = VerilogWire(f"din_relu_{self.id}")
        self.out_wire = VerilogWire(f"dout_relu_{self.id}")
        self.work = deque()
    
    def push_work(self, a):
        self.work.append(f"{self.in_wire} <= {a}")

    def push_read_result(self, r):
            self.work.append(f"{r} <= {self.out_wire}")


def ReLU(*idx):
    if idx not in RELUS:
        RELUS[idx] = RReLU(idx)
    relu = RELUS[idx]

    def op(*args, **kwargs):
        a = args
        relu.push_work(a)

        return relu.out_wire
    
    return op


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
    return any([len(macs) for macs in MAC_QUEUES.values()])


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

    for mac in MACS.values():
        print(make_mac(mac.id), file=FILE)

    for relu in RELUS.values():
        print(make_relu(relu.id), file=FILE)

    for _mac_idx, mac in MACS.items():
        print(indent("always @(posedge ap_clk) begin", "\t"), file=FILE)
        for assign in mac.work:
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
