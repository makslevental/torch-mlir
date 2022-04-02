import queue
import re
import struct
import sys
from collections import defaultdict
import argparse

import numpy as np


def collapse_ariths_llvm(fp):
    """
    %val_145 = udiv i64 12, 121
    %val_146 = urem i64 12, 121
    %val_147 = udiv i64 %val_146, 121
    %val_148 = urem i64 %val_146, 121
    %val_149 = udiv i64 %val_148, 11
    %val_150 = urem i64 %val_148, 11
    %val_151 = getelementptr inbounds [1 x [1 x [11 x [11 x float]]]], [1 x [1 x [11 x [11 x float]]]]* %arg_2, i64 0, i64 %val_145, i64 %val_147, i64 %val_149, i64 %val_150
    """
    csts = {}

    all_sig_lines = []

    with open(fp) as infile:
        for line in infile:
            if "udiv i64" in line or "urem i64" in line:
                ident, instr = map(lambda x: x.strip(), line.split("="))
                op, ops = map(lambda x: x.strip(), instr.split("i64"))
                lhs, rhs = map(lambda x: x.strip(), ops.split(","))
                lhs = int(csts.get(lhs, lhs))
                rhs = int(csts.get(rhs, rhs))

                if "udiv" == op:
                    csts[ident] = str(lhs // rhs)
                elif "urem" == op:
                    csts[ident] = str(lhs % rhs)
                else:
                    raise Exception
            else:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)
                if idents:
                    if "=" in line:
                        idents = idents[1:]

                    for ident in idents:
                        if ident in csts:
                            val = csts[ident]
                            line = re.sub(fr"{ident}\b", f"{val}", line)

                all_sig_lines.append(line)

    with open(fp, "w") as test_file:
        for l in all_sig_lines:
            test_file.write(l)


def collapse_gep(fp):
    "%val_1163887 = getelementptr float, float* %val_65, i64 966"
    "%val_1163382 = load float, float* getelementptr inbounds ([32 x [64 x [1 x [1 x float]]]], [32 x [64 x [1 x [1 x float]]]]* @__constant_32x64x1x1xf32, i64 0, i64 11, i64 62, i64 0, i64 0), align 4"

    back_csts = {}
    first_gep = {}

    all_sig_lines = []

    with open(fp) as infile:
        for line in infile:
            if "= getelementptr" in line and "alloca" not in line:
                ident, instr = map(lambda x: x.strip(), line.split("="))
                if instr in first_gep:
                    back_csts[ident] = first_gep[instr]
                else:
                    first_gep[instr] = ident
                    all_sig_lines.append(line)
            else:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)
                if idents:
                    if "=" in line:
                        idents = idents[1:]

                    for ident in idents:
                        if ident in back_csts:
                            val = back_csts[ident]
                            line = re.sub(fr"{ident}\b", f"{val}", line)

                all_sig_lines.append(line)

    with open(fp, "w") as test_file:
        for l in all_sig_lines:
            test_file.write(l)


def hoist_constants(fp):
    csts = {}
    unique_used_csts = set()
    FUNC_LINE = "llvm.func @forward"
    func_line_number = None

    all_sig_lines = []

    with open(fp) as infile:
        i = -1
        n = 0
        for line in infile:
            i += 1

            if "unreal" in line:
                raise Exception

            if FUNC_LINE in line:
                func_line_number = i

            if "llvm.mlir.constant" in line and "index" in line:
                ident, op = map(lambda x: x.strip(), line.split("="))
                val = re.search(
                    r"llvm.mlir.constant\((\d+) : index\) : i64", op
                ).groups()[0]
                csts[ident] = op, val
            else:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)
                if not idents:
                    all_sig_lines.append(line)
                    continue
                else:
                    if "=" in line:
                        idents = idents[1:]

                    for ident in idents:
                        if ident in csts:
                            _op, val = csts[ident]
                            unique_used_csts.add(val)
                            line = re.sub(fr"\b{ident[1:]}\b", f"cst_{val}", line)

                    all_sig_lines.append(line)

    with open(fp, "w") as test_file:
        for l in all_sig_lines[: func_line_number + 1]:
            test_file.write(l)

        test_file.write("\n\n")

        for u in unique_used_csts:
            test_file.write(f"%cst_{u} = llvm.mlir.constant({u} : index) : i64\n")

        test_file.write("\n\n")

        for l in all_sig_lines[func_line_number + 1 :]:
            test_file.write(l)


def simplify_adds_mults(fp):
    csts = {}
    FUNC_LINE = "llvm.func @forward"
    func_line_number = None

    all_sig_lines = []
    with open(fp) as infile:
        i = -1
        for line in infile:
            i += 1
            if not line:
                continue

            if FUNC_LINE in line:
                func_line_number = i

            if "llvm.mlir.constant" in line and "index" in line:
                ident, op = map(lambda x: x.strip(), line.split("="))
                val = re.search(
                    r"llvm.mlir.constant\((\d+) : index\) : i64", op
                ).groups()[0]
                csts[ident] = op, val
            else:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)
                if not idents:
                    all_sig_lines.append(line)
                    continue
                else:
                    if "=" in line:
                        idents = idents[1:]

                    if "llvm.mul " in line:
                        ident, op = map(lambda x: x.strip(), line.split("="))
                        if all([i in csts for i in idents]):
                            lhs, rhs = map(int, [csts[i][1] for i in idents])

                            val = lhs * rhs
                            op = f"llvm.mlir.constant({val} : index) : i64"
                            csts[ident] = op, str(val)
                        else:
                            all_sig_lines.append(
                                line
                            )  # uses[ident].append((i, lhs, rhs))

                    elif "llvm.add " in line:
                        ident, op = map(lambda x: x.strip(), line.split("="))
                        if all([i in csts for i in idents]):
                            lhs, rhs = map(int, [csts[i][1] for i in idents])
                            val = lhs + rhs
                            op = f"llvm.mlir.constant({val} : index) : i64"
                            csts[ident] = op, str(val)
                        else:
                            all_sig_lines.append(line)
                    else:
                        all_sig_lines.append(line)

    with open(f"{fp}", "w") as test_file:
        for l in all_sig_lines[: func_line_number + 1]:
            if l:
                test_file.write(l)

        test_file.write("\n\n")

        for ident, (op, _val) in csts.items():
            test_file.write(f"{ident} = {op}\n")

        test_file.write("\n\n")

        for l in all_sig_lines[func_line_number + 1 :]:
            if l:
                test_file.write(l)


def remove_dbg(fp):
    all_lines = []
    with open(f"{fp}") as test_file:
        for line in test_file:
            if ", !dbg" in line:
                line = re.sub(r", !dbg !\d+", "", line)
            if "DILocation" not in line:
                all_lines.append(line)

    with open(f"{fp}", "w") as test_file:
        test_file.writelines(all_lines)


def embed_csts(fp):
    csts = {}
    FUNC_LINE = "llvm.func @forward"
    func_line_number = None

    all_sig_lines = []
    with open(fp) as infile:
        i = -1
        for line in infile:
            i += 1
            if not line:
                continue

            if FUNC_LINE in line:
                func_line_number = i

            if "llvm.mlir.constant" in line and "index" in line:
                ident, op = map(lambda x: x.strip(), line.split("="))
                val = re.search(
                    r"llvm.mlir.constant\((\d+) : index\) : i64", op
                ).groups()[0]
                csts[ident] = op, val
            else:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)
                if not idents:
                    all_sig_lines.append(line)
                    continue
                else:
                    if "=" in line:
                        idents = idents[1:]

                    for ident in idents:
                        if ident in csts:
                            line = re.sub(fr"\b{ident[1:]}\b", f"{val}", line)


def double_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def change_floats(fp):
    def gen_fp(x):
        return

    all_sig_lines = []
    with open(fp) as infile:
        for line in infile:
            if "@__constant_" in line and "f32" in line:
                # line = re.sub(r"(\d.\d+e\+\d\d)", lambda x: f"{np.random.uniform(1, 9):.6e}", line)
                line = re.sub(
                    r"(\d.\d+e\+\d\d)",
                    lambda x: f"{double_to_hex(np.random.uniform(1, 9))}00000000".upper().replace(
                        "X", "x"
                    ),
                    line,
                )
            all_sig_lines.append(line)

    with open(f"{fp}", "w") as test_file:
        test_file.writelines(all_sig_lines)


def stores_loads(fp):
    """
      store float %val_248435, float* %val_248436, align 4
      %val_248438 = load float, float* %val_248340, align 4
    """

    loads = defaultdict(list)
    stores = defaultdict(list)
    all_sig_lines = []

    last_alloca_line = -1
    with open(fp) as infile:

        for i, line in enumerate(infile):
            if "alloca" in line:
                last_alloca_line = max(i, last_alloca_line)

            if ("load" in line and "getelementptr" not in line) or "store" in line:
                idents = re.findall(r"(%[\d|a-z|_]*)", line)

                if "load" in line:
                    assign, source = idents
                    loads[source].append((assign, line))
                elif "store" in line and "float*" in line:
                    try:
                        val, assign = idents
                        stores[assign].append((i, val))
                    except Exception as e:
                        print(e)
                        print("line ", i)
                        print(line)
            all_sig_lines.append(line)

    redundant_loads = set()
    canonical_loads_map = {}

    for source, assigns_lines in loads.items():
        if source not in stores:
            if len(assigns_lines) > 1:
                # print("no stores and multiple loads", k, v)
                canonical_assign, _canonical_line = assigns_lines[0]
                for assign, line in assigns_lines[1:]:
                    redundant_loads.add(line.strip())
                    canonical_loads_map[assign] = canonical_assign
        else:
            # print("last store", k, stores[k][-1])
            # print("all loads", k, v)
            pass

    # # TODO: find pairs of stores and loads
    # fifo queue -> each store is a push, each load is a pop
    # for store_assign, (i, val) in stores.items():
    #     # find next load
    #     load_assign, line = next(loads[store_assign])

    with open(fp, "w") as test_file:
        for i, line in enumerate(all_sig_lines):
            if line.strip() in redundant_loads:
                continue

            idents = re.findall(r"(%[\d|a-z|_]*)", line)
            for ident in idents:
                if ident in canonical_loads_map:
                    canonical_assign = canonical_loads_map[ident]
                    line = re.sub(fr"{ident}\b", f"{canonical_assign}", line)

            test_file.write(line)


def drop_init_tensors(fp):
    pat1 = r"""scf\.parallel .*
      %\d+ = memref\.load .*
      memref\.store .*
      scf\.yield
    }
    """
    pat2 = r"""scf.parallel .*
      memref.store %c.*
      scf.yield
    }
    """
    # pat1 = r"""    (%\d+ = memref\.alloca\(\) .*)
    # scf\.parallel .*
    #   %\d+ = memref\.load .*
    #   memref\.store .*
    #   scf\.yield
    # }
    # """
    # pat2 = r"""    (%\d+ = memref\.alloca\(\) .*)
    # scf.parallel .*
    #   memref.store %c.*
    #   scf.yield
    # }
    # """
    with open(fp) as infile:
        str = infile.read()

    sub1 = re.sub(pat1, "\n    ", str, re.MULTILINE)
    sub2 = re.sub(pat2, "\n    ", sub1, re.MULTILINE)
    with open(fp, "w") as infile:
        infile.write(sub2)


def stores_loads2(fp):
    """
      loading from and storing to an arg:

      %val_12 = alloca float, i64 ptrtoint (float* getelementptr (float, float* null, i64 8) to i64), align 4

      %val_19 = getelementptr inbounds [1 x [4 x [2 x [2 x float]]]], [1 x [4 x [2 x [2 x float]]]]* %arg_3, i64 0, i64 0, i64 0, i64 0, i64 0
      %val_20 = load float, float* %val_19, align 4

      ...

      %val_248 = fmul float %val_20, %val_245
      %val_249 = fadd float %val_247, %val_248

      ...

      %val_246 = getelementptr float, float* %val_12, i64 4
      store float %val_249, float* %val_246, align 4

    """

    load_to_ssa = {}
    stores = defaultdict(queue.Queue)
    all_sig_lines = []
    elementptrs_to_args = set()

    with open(fp) as infile:
        for i, line in enumerate(infile):
            idents = re.findall(r"(%[\d|a-z|_]*)", line)

            # allow stores to inout params
            if "define void @forward" in line:
                args = set(idents)
            if "= getelementptr" in line and "@__constant" not in line:
                gep, ssa = idents
                if ssa in args:
                    elementptrs_to_args.add(gep)

            # collect stores that go to canonical ptrs (they're canonical after store_load previous pass
            # assign is a ptr (or address or whatever)
            if "store" in line and "float*" in line:
                val, gep = idents
                # make sure you're still storing to inout args
                if gep not in elementptrs_to_args:
                    stores[gep].put(val)
                    continue
            # find loads that aren't gep loads but load from pointers
            # (e.g., %val_8225 = load float, float* getelementptr inbounds... since those come from constants?)
            # in order to tie them back to the source (i.e., the most recent gep that should've been stored to)
            elif "load" in line and "getelementptr" not in line:
                assign, gep = idents
                if gep in stores and not stores[gep].empty():
                    load_to_ssa[assign] = stores[gep].get()
                    continue

            all_sig_lines.append(line)

    with open(fp, "w") as test_file:
        for i, line in enumerate(all_sig_lines):
            idents = re.findall(r"(%[\d|a-z|_]*)", line)
            for ident in idents:
                if ident in load_to_ssa:
                    canonical_ssa = load_to_ssa[ident]
                    line = re.sub(fr"{ident}\b", f"{canonical_ssa}", line)

            test_file.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("fp")
    parser.add_argument("--drop_init_tensors", action="store_true")

    args = parser.parse_args()
    fp = args.fp
    if args.drop_init_tensors:
        drop_init_tensors(fp)
    else:
        if "vitis" not in fp:
            remove_dbg(fp)
            hoist_constants(fp)
            for i in range(5):
                simplify_adds_mults(fp)
            hoist_constants(fp)
            collapse_ariths_llvm(fp)
        else:
            collapse_ariths_llvm(fp)
            collapse_gep(fp)
            stores_loads(fp)
            stores_loads2(fp)
