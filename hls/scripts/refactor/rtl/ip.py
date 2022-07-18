from textwrap import dedent
from typing import Tuple

from dataclasses import dataclass

from hls.scripts.refactor.ops import OpType
from hls.scripts.refactor.util import remove_all_leading_whitespace


@dataclass(frozen=True)
class Wire:
    id: str
    precision: int

    def __str__(self):
        return f"{self.id}_wire"

    def instantiate(self):
        return f"wire [{self.precision - 1}:0] {self};"


@dataclass(frozen=True)
class Reg:
    id: str
    precision: int

    def __str__(self):
        return f"{self.id}_reg"

    def instantiate(self):
        return f"reg [{self.precision - 1}:0] {self};"


def generate_flopoco_fp(op_type, instance_name, id, x, y, r, keep):
    return dedent(
        f"""\
                    {'(* keep = "true" *) ' if keep else ''}{op_type} #({id}) {instance_name}(
                        .clk(clk),
                        .X({x}),
                        .Y({y}),
                        .R({r})
                    );
                    """
    )


def generate_xilinx_fp(op_type, instance_name, id, precision, a, b, res, keep):
    return f"""\
            {'(* keep = "true" *) ' if keep else ''}{op_type} #({id}, {precision}) {instance_name}(
                .clk(clk),
                .reset(0'b1),
                .clk_enable(1'b1),
                .a({a}),
                .b({b}),
                .res({res})
            );
            """


IP_ID = 0


class IP:
    def __init__(
        self, op_type: OpType, pe_idx: Tuple[int, ...], precision: int, keep=True
    ):
        global IP_ID
        IP_ID += 1

        self.id = IP_ID
        self.op_type = op_type.value
        self.pe_idx_str = "_".join(map(str, pe_idx))
        self.precision = precision
        self.keep = keep
        self.instance_name = f"{self.op_type}_{self.pe_idx_str}"


class FAddOrMulIP(IP):
    def __init__(
        self, op_type: OpType, pe_idx: Tuple[int, ...], precision: int, keep=True
    ):
        super().__init__(op_type, pe_idx, precision, keep)
        self.x = Reg(f"{self.instance_name}_x", precision)
        self.y = Reg(f"{self.instance_name}_y", precision)
        self.r = Wire(f"{self.instance_name}_r", precision)

    def instantiate(self):
        wires_regs = remove_all_leading_whitespace(
            f"""\
                {self.x.instantiate()}
                {self.y.instantiate()}
                {self.r.instantiate()}
            """
        )
        instance = generate_flopoco_fp(
            self.op_type, self.instance_name, self.id, self.x, self.y, self.r, self.keep
        )
        return wires_regs + instance


class FAdd(FAddOrMulIP):
    def __init__(self, pe_idx, precision):
        super().__init__(OpType.ADD, pe_idx, precision)


class FMul(FAddOrMulIP):
    def __init__(self, pe_idx, precision):
        super().__init__(OpType.MUL, pe_idx, precision)


def generate_relu_or_neg(op_type, id, precision, instance_name, a, res):
    op = dedent(
        f"""\
            {op_type} #({id}, {precision}) {instance_name}(
                .a({a}),
                .res({res})
            );
        """
    )
    return op


class ReLUOrNegIP(IP):
    def __init__(
        self, op_type: OpType, pe_idx: Tuple[int, ...], precision: int, keep=True
    ):
        super().__init__(op_type, pe_idx, precision, keep)
        self.a = Reg(f"{self.instance_name}_a", precision)
        self.res = Wire(f"{self.instance_name}_res", precision)

    def instantiate(self):
        wires_regs = remove_all_leading_whitespace(
            f"""\
                {self.a.instantiate()}
                {self.res.instantiate()}
            """
        )
        instance = generate_relu_or_neg(
            self.op_type, self.id, self.precision, self.instance_name, self.a, self.res
        )
        return wires_regs + instance


class ReLU(ReLUOrNegIP):
    def __init__(self, pe_idx, precision):
        super().__init__(OpType.RELU, pe_idx, precision)


class Neg(ReLUOrNegIP):
    def __init__(self, pe_idx, precision):
        super().__init__(OpType.NEG, pe_idx, precision)


def make_shift_rom(precision, id, op_name, data_out_wire, addr_width):
    res = []
    # waddr = ','.join([f"0'b0"] * addr_width)
    # write_data = ','.join([f"0'b0"] * precision)
    res.append(
        dedent(
            f"""\
            reg[{addr_width}-1:0] {op_name}_raddr;
            always @(posedge clk) begin
                if ({op_name}_raddr == RAM_SIZE) begin
                    {op_name}_raddr = 0;
                end else begin
                    {op_name}_raddr <= {op_name}_raddr+1'b1;
                end
            end
        
            wire   [{precision - 1}:0] {data_out_wire};
            simple_dual_rw_ram #({id}, {precision}, {2 ** addr_width}) {op_name}(
                .wclk(clk),
                .waddr({addr_width}'b0),
                .write_data({op_name}_raddr),
                .write_en(1'b1),
                .rclk(clk),
                .raddr({op_name}_raddr),
                .read_data({data_out_wire})
            );
            """
        )
    )
    return "\n".join(res)


if __name__ == "__main__":
    print(FAdd((0, 0), 11).instantiate())
    print(FMul((0, 0), 11).instantiate())
    print(ReLU((0, 0), 11).instantiate())
    print(Neg((0, 0), 11).instantiate())
