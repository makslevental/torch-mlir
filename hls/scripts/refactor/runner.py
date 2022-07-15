import inspect
import io
import itertools
import logging

from hls.scripts.refactor.memref import MemRef, GlobalMemRef
from hls.scripts.refactor.val import make_latency_attrs
from hls.scripts.refactor.state import state as State, logger


def extend_idx(pe_idx):
    idx = pe_idx[:]
    if len(pe_idx) < 4:
        idx = 4 * [0]
        idx[-len(pe_idx) :] = pe_idx
    pe_idx = tuple(idx)
    return pe_idx


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def make_args_globals(args):
    inputs = []
    output = []
    globals = []
    for _arg_name, arg in args.items():
        if isinstance(arg, MemRef):
            if arg.input:
                inputs.extend(arg.val_names)
            elif arg.output:
                output.append(arg)
        elif isinstance(arg, GlobalMemRef):
            globals.extend(arg.val_names)

    assert len(output) == 1
    output = output[0]
    return inputs, globals, output


def MLIRForward(args, forward):
    inputs, globals, output = make_args_globals(args)
    output_dtypes = ", ".join([State.dtype] * output.numel)
    inps_globals = [f"{v}: {State.dtype}" for v in inputs + globals]
    State.emit(
        f"func.func @forward({', '.join(inps_globals)}) -> ({output_dtypes})\n",
    )
    State.emit(make_latency_attrs())

    OLD_FILE = State.swap_output_file(io.StringIO())
    forward()

    logger.debug(f"num unique pes {State.num_unique_pes}")

    OLD_FILE.write(State.read_output_file())
    State.swap_output_file(OLD_FILE)

    State.emit(f"return {', '.join(output.val_names)}: {output_dtypes}")
    State.emit("}")


def parfor(ranges):
    def wrapper(body):
        for i, idx in enumerate(itertools.product(*ranges)):
            State.pe_idx = extend_idx(idx)
            body(*idx)

    return wrapper


def Forward(forward, fp):
    State.set_output_file(fp)
    args = get_default_args(forward)
    MLIRForward(args, forward)
