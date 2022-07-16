from hls.scripts.refactor.state import state as State


def Alias(dst: "MemRef", src: "MemRef"):
    from hls.scripts.refactor.memref import MemRef

    assert isinstance(src, MemRef)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy


class FMAC:
    def __init__(self, *pe_idx):
        from hls.scripts.refactor.runner import extend_idx

        pe_idx = extend_idx(pe_idx)
        self.pe_idx = pe_idx
        State.debug_print(f"MAC {pe_idx} starts")
        self.most_recent_add = None
        self.most_recent_mul = None

    def _collapse(self, prev_res, a, b):
        from hls.scripts.refactor.val import create_new_op, OpType

        prev_op = State.get_arg_src(prev_res)
        op, res = create_new_op(OpType.MUL, pe_idx=prev_op.pe_idx, extra_attrs=(("mac_reg", str(prev_res)),))
        State.emit(f"{res} = {str(op).replace('ARGS', f'{a}, {b}')}")

    def Add(self, a, b):
        if self.most_recent_add is None or not State.collapse_macs:
            self.most_recent_add = a + b
        else:
            self._collapse(self.most_recent_add, a, b)
        return self.most_recent_add

    def Mul(self, a, b):
        from hls.scripts.refactor.val import create_new_op, create_new_op_arg, OpType

        if self.most_recent_mul is None or not State.collapse_macs:
            op, op_res = create_new_op(OpType.MUL, add_aux_dep=True)
            op_str = str(op)
            a = str(create_new_op_arg(op, a, op_res))
            b = str(create_new_op_arg(op, b, op_res))
            State.emit(
                f"{op_res} = {op_str.replace('ARGS', ', '.join([a, b]))}"
            )
            self.most_recent_mul = op_res
        else:
            self._collapse(self.most_recent_mul, a, b)
        return self.most_recent_mul

    def Result(self):
        if State.collapse_macs:
            State.debug_print(f"MAC {self.pe_idx} ends")
            from hls.scripts.refactor.val import Copy

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
