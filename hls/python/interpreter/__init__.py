import os
from typing import Tuple

if os.environ.get("VAL_FACTORY", "CPPVal") == "CPPVal":
    from hls.python.interpreter.cpp_val import (
        Array,
        GlobalArray,
        ParFor,
        Copy,
        Forward,
        FMac,
        Add,
        ReLU,
        ReduceAdd,
        FMulAdd
    )
elif os.environ.get("VAL_FACTORY") == "LLVMVal":
    from hls.python.interpreter.llvm_val import *
elif os.environ.get("VAL_FACTORY") == "VerilogVal":
    from hls.python.interpreter.verilog_val import *
else:
    raise Exception("unknown evaluator")

ArrayIndex = Tuple[int]

# __all__ = module1.__all__ + module2.__all__

__all__ = [
    "Array",
    "GlobalArray",
    "ParFor",
    "Copy",
    "Forward",
    "FMac",
    "Add",
    "ReLU",
    "ReduceAdd",
    "FMulAdd"
]
