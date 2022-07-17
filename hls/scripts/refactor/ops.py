from hls.scripts.refactor.state import state as State, idx_to_str
from hls.scripts.refactor.val import OpType, overload_op, create_new_op


def Alias(dst: "MemRef", src: "MemRef"):
    from hls.scripts.refactor.memref import MemRef

    assert isinstance(src, MemRef)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy


class FMAC:
    from hls.scripts.refactor.runner import extend_idx

    def __init__(self, *pe_idx):
        pe_idx = FMAC.extend_idx(pe_idx)
        self.pe_idx = pe_idx
        self.res_reg = idx_to_str(pe_idx)
        State.debug_print(f"MAC {pe_idx} starts")
        self.most_recent_add = None
        self.most_recent_mul = None

    def _collapse(self, op_type, prev_res, a, b):
        prev_op = State.get_arg_src(prev_res)
        res = create_new_op(op_type, (a, b), pe_idx=prev_op.pe_idx, res_reg=f"{op_type.value}_{self.res_reg}")
        return res

    def Add(self, a, b):
        if self.most_recent_add is None or not State.collapse_macs:
            self.most_recent_add = a + b
        else:
            self.most_recent_add = self._collapse(
                OpType.ADD, self.most_recent_add, a, b
            )
        return self.most_recent_add

    def Mul(self, a, b):
        if self.most_recent_mul is None or not State.collapse_macs:
            self.most_recent_mul = create_new_op(
                OpType.MUL, (a, b), add_aux_dep=True
            )
        else:
            self.most_recent_mul = self._collapse(
                OpType.MUL, self.most_recent_mul, a, b
            )
        return self.most_recent_mul

    def Result(self):
        if State.collapse_macs:
            State.debug_print(f"MAC {self.pe_idx} ends")

            op_res = Copy(self.most_recent_add)
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


ReLU = lambda x: overload_op(OpType.RELU)(x)
Copy = lambda x: overload_op(OpType.COPY)(x)
