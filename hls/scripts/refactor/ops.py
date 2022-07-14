from hls.scripts.refactor import state as state


def Alias(dst: "MemRef", src: "MemRef"):
    from hls.scripts.refactor.memref import MemRef

    assert isinstance(src, MemRef)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy


def Constant(val):
    from hls.scripts.refactor.val import Val

    value = Val(str(val))
    state.VAL_SOURCE[value] = state.CONSTANT
    state.emit(f"{value} = arith.constant {val} : {state.DTYPE}")
    return value


def ReLU(arg):
    from hls.scripts.refactor.val import create_new_op, OpType, create_new_op_arg

    op, v = create_new_op(OpType.RELU, state.PE_IDX)
    arg = create_new_op_arg(arg, op, v)
    op_str = str(op)
    state.emit(f"{v} = {op_str.replace('ARGS', str(arg))}")
    return v


def Copy(arg):
    from hls.scripts.refactor.val import create_new_op, OpType, create_new_op_arg

    op, v = create_new_op(OpType.COPY, state.PE_IDX)
    arg = create_new_op_arg(arg, op, v)
    op_str = str(op)
    state.emit(f"{v} = {op_str.replace('ARGS', str(arg))}")
    return v


class FMAC:
    def __init__(self, *pe_idx):
        from hls.scripts.refactor.runner import extend_idx

        pe_idx = extend_idx(pe_idx)
        self.pe_idx = pe_idx
        state.emit(f"// MAC {pe_idx} starts")
        self.most_recent_add = None
        self.most_recent_mul = None

    def Add(self, a, b):
        if self.most_recent_add is None or not state.COLLAPSE_MACS:
            self.most_recent_add = a + b
        else:
            op_str = str(state.VAL_SOURCE[self.most_recent_add])
            state.emit(
                f"{self.most_recent_add} = {op_str.replace('ARGS', f'{a}, {b}')}"
            )
        return self.most_recent_add

    def Mul(self, a, b):
        if self.most_recent_mul is None or not state.COLLAPSE_MACS:
            self.most_recent_mul = a * b
        else:
            op_str = str(state.VAL_SOURCE[self.most_recent_mul])
            state.emit(
                f"{self.most_recent_mul} = {op_str.replace('ARGS', f'{a}, {b}')}"
            )
        return self.most_recent_mul

    def Result(self):
        if state.COLLAPSE_MACS:
            state.emit(f"// MAC {self.pe_idx} ends")
            from hls.scripts.refactor.val import (
                create_new_op,
                OpType,
                create_new_op_arg,
            )

            op, v = create_new_op(OpType.COPY, state.PE_IDX)
            arg = create_new_op_arg(self.most_recent_add, op, v)
            op_str = str(op)
            state.emit(f"{v} = {op_str.replace('ARGS', str(arg))}")
        else:
            v = self.most_recent_add
        return v


def ReduceAdd(src_arr: "MemRef", dst_arr: "MemRef"):
    prev_sums = list(src_arr.registers.values())
    while len(prev_sums) > 1:
        next_sums = []
        while len(prev_sums) > 1:
            left = prev_sums.pop()
            state.PE_IDX = state.VAL_TO_PE_IDX[left]
            next_sums.append(left + prev_sums.pop())
        if len(prev_sums) == 1:
            left = next_sums[-1]
            state.PE_IDX = state.VAL_TO_PE_IDX[left]
            next_sums[-1] = left + prev_sums[0]
        prev_sums = next_sums
    dst_arr[
        0,
    ] = prev_sums[0]