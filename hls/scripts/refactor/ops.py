from hls.scripts.refactor.state import state as State


def Alias(dst: "MemRef", src: "MemRef"):
    from hls.scripts.refactor.memref import MemRef

    assert isinstance(src, MemRef)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy


def Constant(val):
    from hls.scripts.refactor.val import Val

    value = Val(str(val))
    State.add_constant(value)
    State.emit(f"{value} = arith.constant {val} : {State.dtype}")
    return value


def ReLU(arg):
    from hls.scripts.refactor.val import create_new_op, OpType, create_new_op_arg

    op, op_res = create_new_op(OpType.RELU)
    arg = create_new_op_arg(op, arg, op_res)
    op_str = str(op)
    State.emit(f"{op_res} = {op_str.replace('ARGS', str(arg))}")
    return op_res


def Copy(arg):
    from hls.scripts.refactor.val import create_new_op, OpType, create_new_op_arg

    op, op_res = create_new_op(OpType.COPY)
    arg = create_new_op_arg(op, arg, op_res)
    op_str = str(op)
    State.emit(f"{op_res} = {op_str.replace('ARGS', str(arg))}")
    return op_res


class FMAC:
    def __init__(self, *pe_idx):
        from hls.scripts.refactor.runner import extend_idx

        pe_idx = extend_idx(pe_idx)
        self.pe_idx = pe_idx
        State.debug_print(f"MAC {pe_idx} starts")
        self.most_recent_add = None
        self.most_recent_mul = None

    def Add(self, a, b):
        if self.most_recent_add is None or not State.collapse_macs:
            self.most_recent_add = a + b
        else:
            op_str = str(State.get_arg_src(self.most_recent_add))
            State.emit(
                f"{self.most_recent_add} = {op_str.replace('ARGS', f'{a}, {b}')}"
            )
        return self.most_recent_add

    def Mul(self, a, b):
        if self.most_recent_mul is None or not State.collapse_macs:
            self.most_recent_mul = a * b
        else:
            op_str = str(State.get_arg_src(self.most_recent_mul))
            State.emit(
                f"{self.most_recent_mul} = {op_str.replace('ARGS', f'{a}, {b}')}"
            )
        return self.most_recent_mul

    def Result(self):
        if State.collapse_macs:
            State.debug_print(f"MAC {self.pe_idx} ends")
            from hls.scripts.refactor.val import (
                create_new_op,
                OpType,
                create_new_op_arg,
            )

            op, op_res = create_new_op(OpType.COPY)
            arg = create_new_op_arg(op, self.most_recent_add, op_res)
            op_str = str(op)
            State.emit(f"{op_res} = {op_str.replace('ARGS', str(arg))}")
        else:
            op_res = self.most_recent_add
        return op_res


def ReduceAdd(vals):
    prev_sums = list(vals)
    while len(prev_sums) > 1:
        next_sums = []
        while len(prev_sums) > 1:
            left = prev_sums.pop()
            State.pe_idx = State.get_arg_src(left).pe_idx
            next_sums.append(left + prev_sums.pop())
        if len(prev_sums) == 1:
            left = next_sums[-1]
            State.pe_idx = State.get_arg_src(left).pe_idx
            next_sums[-1] = left + prev_sums[0]
        prev_sums = next_sums
    return prev_sums[0]