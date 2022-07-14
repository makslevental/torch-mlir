import inspect
import io
import itertools

from hls.scripts.refactor.memref import MemRef, GlobalMemRef
from hls.scripts.refactor.val import make_latency_attrs
import hls.scripts.refactor.state as state


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
    output_dtypes = ", ".join([state.DTYPE] * output.numel)
    inps_globals = [f"{v}: {state.DTYPE}" for v in inputs + globals]
    state.emit(
        f"func.func @forward({', '.join(inps_globals)}) -> ({output_dtypes})\n",
    )
    state.emit(make_latency_attrs())

    OLD_FILE = state.OUTPUT_FILE
    state.OUTPUT_FILE = io.StringIO()
    forward()

    print(f"num unique pes {len(state.UNIQUE_IDXS)}")

    state.OUTPUT_FILE.seek(0)
    OLD_FILE.write(state.OUTPUT_FILE.read())

    print(f"return {', '.join(output.val_names)}: {output_dtypes}", file=OLD_FILE)
    print("}", file=OLD_FILE)

    OLD_FILE.close()


def ParFor(body, ranges):
    global IDX
    for i, idx in enumerate(itertools.product(*ranges)):
        IDX = extend_idx(idx)
        body(*idx, state.OUTPUT_FILE)


def Forward(forward):
    args = get_default_args(forward)
    MLIRForward(args, forward)
