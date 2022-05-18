import io
import itertools
import json
import os
from collections import deque


def format_cst(cst):
    return cst
    # return float(np.random.randint(1, 100000))
    # return cst
    # return (handleDoubleToHex(cst)[:11] + "0" * 7).upper().replace("X", "x")


VAR_COUNT = 0

FILE = open("forward.cpp", "w")


class CPPVal:
    def __init__(self, name):
        global VAR_COUNT
        self.name = f"{name}"
        self.var_id = f"val_{VAR_COUNT}"
        VAR_COUNT += 1

    def __mul__(self, other):
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(* ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"float {v} = fmul(1.0, {other});", file=FILE)
        elif "-1" in f"{other}":
            print(f"float {v} = fmul({self}, 1.0);", file=FILE)
        else:
            print(f"float {v} = fmul({self}, {other});", file=FILE)
        return v

    def __add__(self, other):
        # <result> = fadd float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(+ ({self}) ({other}))")
        if "-1" in f"{self}":
            print(f"float {v} = (0.0 + {other});", file=FILE)
        elif "-1" in f"{other}":
            print(f"float {v} = ({self} + 0.0);", file=FILE)
        else:
            print(f"float {v} = ({self} + {other});", file=FILE)
        return v

    def __sub__(self, other):
        # <result> = fsub float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(- ({self}) ({other}))")
        print(f"float {v} = fsub({self}, {other});", file=FILE)
        return v

    def __truediv__(self, other):
        # <result> = fdiv float 4.0, %var
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(/ ({self}) ({other}))")
        print(f"float {v} = fdiv({self}, {other});", file=FILE)
        return v

    def __floordiv__(self, other):
        raise Exception("wtfbbq")

    def __gt__(self, other):
        # <result> = fcmp ugt float 4.0, 5.0
        if isinstance(other, (float, int, bool)):
            other = CPPConstant(other)
        v = CPPVal(f"(> ({self}) ({other}))")
        print(f"float {v} = fcmpugt({self}, {other});", file=FILE)
        return v

    def __str__(self):
        return f"{self.var_id}"


class ArrayVal(CPPVal):
    def __str__(self):
        return f"{self.name}{get_array_type(self.var_id, ptr=False, nd=True)}"


class GlobalVal(CPPVal):
    def __str__(self):
        return f"{self.name}{get_array_type(self.var_id, ptr=False, nd=True)}"


class CPPConstant(CPPVal):
    def __init__(self, name):
        super(CPPConstant, self).__init__(name)
        self._fmt = f"{format_cst(self.name)}"

    def __str__(self):
        return self._fmt


class MExp:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, arg):
        if isinstance(arg, (float, int, bool)):
            arg = CPPConstant(arg)
        if isinstance(arg, GlobalVal):
            self.csts.append(arg.var_id)

        if self.output is None:
            self.output = CPPVal(f"(fexp ({arg})")
            print(f"float {self.output};", file=FILE)
        print(f"fexp_{self.id}({arg}, &{self.output});", file=FILE)

        return self.output


EXPS = {}


def Exp(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in EXPS:
        EXPS[idx] = MExp(idx)
    mac = EXPS[idx]

    def op(arg):
        return mac(arg)

    return op


class MReLU:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, arg):
        if isinstance(arg, (float, int, bool)):
            arg = CPPConstant(arg)
        if isinstance(arg, GlobalVal):
            self.csts.append(arg.var_id)

        if self.output is None:
            self.output = CPPVal(f"(freLU ({arg})")
            print(f"float {self.output};", file=FILE)
        print(f"frelu_{self.id}({arg}, &{self.output});", file=FILE)

        return self.output


ReLUS = {}


def ReLU(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in ReLUS:
        ReLUS[idx] = MReLU(idx)
    mac = ReLUS[idx]

    def op(arg):
        return mac(arg)

    return op



class MMul:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = CPPConstant(v)
            if isinstance(v, GlobalVal):
                self.csts.append(v.var_id)

        a, b = args
        if a.name.startswith("_arg"):
            _, arg_name, _ = a.name.split("_", 2)
            arg_name = f"_{arg_name}"
            idx = get_array_type(a.var_id, nd=True)
            a = f"{arg_name}{idx}"

        if isinstance(b, GlobalVal):
            b = len(self.csts)

        if self.output is None:
            self.output = CPPVal(f"(fmul ({a} {b}))")
            print(f"float {self.output};", file=FILE)
        print(f"fmul_{self.id}({a}, {b}, &{self.output});", file=FILE)

        return self.output


MULS = {}


def Mul(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in MULS:
        MULS[idx] = MMul(idx)
    mac = MULS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


class MDiv:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = CPPConstant(v)
            if isinstance(v, GlobalVal):
                self.csts.append(v.var_id)

        a, b = args
        if a.name.startswith("_arg"):
            _, arg_name, _ = a.name.split("_", 2)
            arg_name = f"_{arg_name}"
            idx = get_array_type(a.var_id, nd=True)
            a = f"{arg_name}{idx}"

        if isinstance(b, GlobalVal):
            b = len(self.csts)

        if self.output is None:
            self.output = CPPVal(f"(fdiv ({a} {b})")
            print(f"float {self.output};", file=FILE)
        print(f"fdiv_{self.id}({a}, {b}, &{self.output});", file=FILE)

        return self.output


DIVS = {}


def Div(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in DIVS:
        DIVS[idx] = MDiv(idx)
    mac = DIVS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


class MUgt:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = CPPConstant(v)
            if isinstance(v, GlobalVal):
                self.csts.append(v.var_id)

        a, b = args
        if a.name.startswith("_arg"):
            _, arg_name, _ = a.name.split("_", 2)
            arg_name = f"_{arg_name}"
            idx = get_array_type(a.var_id, nd=True)
            a = f"{arg_name}{idx}"

        if isinstance(b, GlobalVal):
            b = len(self.csts)

        if self.output is None:
            self.output = CPPVal(f"(fugt ({a} {b})")
            print(f"float {self.output};", file=FILE)
        print(f"fugt_{self.id}({a}, {b}, &{self.output});", file=FILE)

        return self.output


UGTS = {}


def Ugt(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in UGTS:
        UGTS[idx] = MUgt(idx)
    mac = UGTS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


class MSub:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = CPPConstant(v)
            if isinstance(v, GlobalVal):
                self.csts.append(v.var_id)

        a, b = args
        if a.name.startswith("_arg"):
            _, arg_name, _ = a.name.split("_", 2)
            arg_name = f"_{arg_name}"
            idx = get_array_type(a.var_id, nd=True)
            a = f"{arg_name}{idx}"

        if isinstance(b, GlobalVal):
            b = len(self.csts)

        if self.output is None:
            self.output = CPPVal(f"(fsub ({a} {b})")
            print(f"float {self.output};", file=FILE)
        print(f"fsub_{self.id}({a}, {b}, &{self.output});", file=FILE)

        return self.output


SUBS = {}


def Sub(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in SUBS:
        SUBS[idx] = MSub(idx)
    mac = SUBS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


class MAdd:
    def __init__(self, idx):
        self.mac_idx = idx
        self.id = "_".join(map(str, idx))
        self.work = deque()
        self.csts = []
        self.output = None

    def __call__(self, *args, **kwargs):
        args = list(args)
        for i, v in enumerate(args):
            if isinstance(v, (float, int, bool)):
                args[i] = CPPConstant(v)
            # if isinstance(v, GlobalVal):
            #     self.csts.append(v.var_id)

        a, b = args
        if a.name.startswith("_arg"):
            _, arg_name, _ = a.name.split("_", 2)
            arg_name = f"_{arg_name}"
            idx = get_array_type(a.var_id, nd=True)
            a = f"{arg_name}{idx}"

        # if isinstance(b, GlobalVal):
        #     b = len(self.csts)

        if self.output is None:
            self.output = CPPVal(f"(fadd ({a} {b})")
            print(f"float {self.output};", file=FILE)
        print(f"fadd_{self.id}({a}, {b}, &{self.output});", file=FILE)

        return self.output


ADDS = {}


def Add(*idx):
    if len(idx) < 4:
        _idx = 4 * [0]
        _idx[0: len(idx)] = idx
        idx = tuple(_idx)

    if idx not in ADDS:
        ADDS[idx] = MAdd(idx)
    mac = ADDS[idx]

    def op(*args, **kwargs):
        return mac(*args, **kwargs)

    return op


def ParFor(body, ranges):
    for i, idx in enumerate(itertools.product(*ranges)):
        body(*idx)


def get_array_type(shape, ptr=True, nd=False):
    if nd:
        typ = ""
        for s in shape:
            typ += f"[{s}]"
    else:
        typ = f"[{shape}]"
    return typ


def make_args_globals(Args):
    from mlir_ops import ArrayDecl, Global

    args = []
    globals = []
    for _arg_name, arg in Args.items():
        if isinstance(arg, ArrayDecl):
            typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
            args.append(f"float {arg.var_name}{typ}")
        elif isinstance(arg, Global):
            typ = get_array_type(arg.curr_shape, nd=True, ptr=False)
            globals.append(f"float {arg.var_name}{typ}")
            # globals.append(arg)

    return args, globals


def make_fmul_json(id):
    return {
        "c_function_name": f"fmul_{id}",
        "rtl_top_module_name": f"fmul_{id}",
        "c_files": [
            {
                "c_file": f"jsons/fmul_{id}.cpp",
                "cflag": ""
            }
        ],
        "rtl_files": [
            f"jsons/fmul_{id}.v"
        ],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "a1"
                }
            },
            {
                "c_name": "idx",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "idx"
                }
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {
                    "data_write_out": "z1",
                    "data_write_valid": "z1_ap_vld"
                }
            }
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue"
        },
        "rtl_performance": {
            "latency": "2",
            "II": "1"
        },
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1"
        }
    }


def make_fadd_json(id):
    return {
        "c_function_name": f"fadd_{id}",
        "rtl_top_module_name": f"fadd_{id}",
        "c_files": [
            {
                "c_file": f"jsons/fadd_{id}.cpp",
                "cflag": ""
            }
        ],
        "rtl_files": [
            f"jsons/fadd_{id}.v"
        ],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "a1"
                }
            },
            {
                "c_name": "b1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "b1"
                }
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {
                    "data_write_out": "z1",
                    "data_write_valid": "z1_ap_vld"
                }
            }
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue"
        },
        "rtl_performance": {
            "latency": "2",
            "II": "1"
        },
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1"
        }
    }


def make_op_json(op_name, id):
    return {
        "c_function_name": f"f{op_name}_{id}",
        "rtl_top_module_name": f"f{op_name}_{id}",
        "c_files": [
            {
                # "c_file": f"jsons/total_f{op_name}.cpp",
                "c_file": f"jsons/f{op_name}_{id}.cpp",
                "cflag": ""
            }
        ],
        "rtl_files": [
            f"jsons/f{op_name}_{id}.v"
            # f"jsons/total_f{op_name}.v"
        ],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "a1"
                }
            },
            {
                "c_name": "b1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "b1"
                }
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {
                    "data_write_out": "z1",
                    "data_write_valid": "z1_ap_vld"
                }
            }
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue"
        },
        "rtl_performance": {
            "latency": "2",
            "II": "1"
        },
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1"
        }
    }

def make_single_arg_op_json(op_name, id):
    return {
        "c_function_name": f"f{op_name}_{id}",
        "rtl_top_module_name": f"f{op_name}_{id}",
        "c_files": [
            {
                "c_file": f"jsons/f{op_name}_{id}.cpp",
                # "c_file": f"jsons/total_f{op_name}.cpp",
                "cflag": ""
            }
        ],
        "rtl_files": [
            f"jsons/f{op_name}_{id}.v"
            # f"jsons/total_f{op_name}.v"
        ],
        "c_parameters": [
            {
                "c_name": "a1",
                "c_port_direction": "in",
                "rtl_ports": {
                    "data_read_in": "a1"
                }
            },
            {
                "c_name": "z1",
                "c_port_direction": "out",
                "rtl_ports": {
                    "data_write_out": "z1",
                    "data_write_valid": "z1_ap_vld"
                }
            }
        ],
        "rtl_common_signal": {
            "module_clock": "ap_clk",
            "module_reset": "ap_rst",
            "module_clock_enable": "ap_ce",
            "ap_ctrl_chain_protocol_idle": "ap_idle",
            "ap_ctrl_chain_protocol_start": "ap_start",
            "ap_ctrl_chain_protocol_ready": "ap_ready",
            "ap_ctrl_chain_protocol_done": "ap_done",
            "ap_ctrl_chain_protocol_continue": "ap_continue"
        },
        "rtl_performance": {
            "latency": "2",
            "II": "1"
        },
        "rtl_resource_usage": {
            "FF": "0",
            "LUT": "0",
            "BRAM": "1",
            "URAM": "0",
            "DSP": "1"
        }
    }


def make_llvm_2_arg_op_prototype(op_name, id):
    return f"""
define void @_Z12f{op_name}_{id}ffPf(float %a1, float %b1, float* %z1) #0 {{
entry:
  call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
  call void (...) @_ssdm_op_BlackBox(float %a1, float %b1, float* %z1)
  call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
  ret void
}}
"""


def make_llvm_1_arg_op_prototype(op_name, id):
    return f"""
define void @_Z12f{op_name}_{id}ffPf(float %a1, float* %z1) #0 {{
entry:
  call void (...) @_ssdm_InlineSelf(i64 2, [1 x i8]* @1)
  call void (...) @_ssdm_op_BlackBox(float %a1, float* %z1)
  call void (...) @_ssdm_op_SpecIPCore(i32 0, i32 580, i32 0, i32 -1)
  ret void
}}
"""

def make_llvm_prefix():
    return """
@0 = private unnamed_addr constant [10 x i8] c"ap_memory\\00"
@1 = private unnamed_addr constant [1 x i8] zeroinitializer
@2 = private unnamed_addr constant [8 x i8] c"forward\\00"

declare void @_ssdm_op_BlackBox(...)

declare void @_ssdm_op_SpecIPCore(...)

declare void @_ssdm_op_SpecInterface(...)

declare void @_ssdm_op_SpecBitsMap(...)

declare void @_ssdm_InlineSelf(...)

declare void @_ssdm_op_SpecTopModule(...)
"""


def CPPForward(Args, OUTPUT_ARRAYS, forward):
    global FILE

    args, globals = make_args_globals(Args)
    llvm_ir = open("forward.ll", "w")
    llvm_ir.write(make_llvm_prefix())
    OLD_FILE = FILE
    FILE = io.StringIO()
    print('extern "C" ' + f"void forward({', '.join(args)}) {{\n", file=FILE)
    forward()

    os.makedirs("jsons", exist_ok=True)
    for op_name, op_dict in [("mul", MULS), ("add", ADDS), ("div", DIVS), ("sub", SUBS), ("ugt", UGTS)]:
        template = open(f"f{op_name}.v").read()
        for idx, op in op_dict.items():
            s = f"void f{op_name}_{op.id}(float a1, float b1, float* z1)"
            print(f"{s};", file=OLD_FILE)

            fmul_cpp = open(f"jsons/f{op_name}_{op.id}.cpp", "w")
            fmul_v = open(f"jsons/f{op_name}_{op.id}.v", "w")
            fmul_cpp.write(f"{s} {{}}\n")
            json.dump(make_op_json(op_name, op.id), open(f"jsons/f{op_name}_{op.id}.json", "w"))
            fmul_v.write(template.replace("XXX", op.id))

            llvm_ir.write(make_llvm_2_arg_op_prototype(op_name, op.id))

    for op_name, op_dict in [("relu", ReLUS), ("exp", EXPS)]:
        template = open(f"f{op_name}.v").read()
        for idx, op in op_dict.items():
            s = f"void f{op_name}_{op.id}(float a1, float* z1)"
            print(f"{s};", file=OLD_FILE)

            fmul_cpp = open(f"jsons/f{op_name}_{op.id}.cpp", "w")
            fmul_v = open(f"jsons/f{op_name}_{op.id}.v", "w")
            fmul_cpp.write(f"{s} {{}}\n")
            json.dump(make_single_arg_op_json(op_name, op.id), open(f"jsons/f{op_name}_{op.id}.json", "w"))
            fmul_v.write(template.replace("XXX", op.id))

            llvm_ir.write(make_llvm_1_arg_op_prototype(op_name, op.id))


    FILE.seek(0)
    OLD_FILE.write(FILE.read())

    for arr in OUTPUT_ARRAYS:
        for index, value in arr.registers.items():
            print(f"{arr.var_name}{get_array_type(index, nd=True)} = {value};", file=OLD_FILE)

    print("return;", file=OLD_FILE)
    print("}", file=OLD_FILE)

# declare void @_ssdm_op_SpecResource(...)
#
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v85, i64 666, i64 23, i64 2)
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v86, i64 666, i64 20, i64 -1)
# call void (...) @_ssdm_op_SpecResource([1 x [3 x [34 x [34 x i8]]]]* %v87, i64 666, i64 22, i64 2)
#
# volatile int8_t v85[1][3][34][34];	// L71
# #pragma HLS BIND_STORAGE variable=v85 type=RAM_2P impl=LUTRAM latency=2
# int8_t v86[1][3][34][34];	// L71
# #pragma HLS BIND_STORAGE variable=v86 type=RAM_1P impl=URAM
# int8_t v87[1][3][34][34];	// L7
# #pragma HLS BIND_STORAGE variable=v87 type=RAM_2P impl=BRAM latency=2V
