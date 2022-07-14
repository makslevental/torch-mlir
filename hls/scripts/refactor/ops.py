from hls.scripts.refactor.memref import MemRef


def Copy(dst: MemRef, src: MemRef):
    assert isinstance(src, MemRef)
    registers_to_copy = src.registers
    dst.registers = registers_to_copy
