import re
import struct
import sys
from collections import defaultdict


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

    with open(f"{fp}.cse", "w") as test_file:
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
                            all_sig_lines.append(line)
                            # uses[ident].append((i, lhs, rhs))

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


import numpy as np

def double_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

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
                    lambda x: f"{double_to_hex(np.random.uniform(1, 9))}00000000".upper().replace("X", "x"),
                    line,
                )
            all_sig_lines.append(line)

    with open(f"{fp}", "w") as test_file:
        test_file.writelines(all_sig_lines)


if __name__ == "__main__":
    fp = sys.argv[1]
    remove_dbg(fp)
    hoist_constants(fp)
    for i in range(5):
        simplify_adds_mults(f"{fp}.cse")
    hoist_constants(f"{fp}.cse")
    # change_floats("/home/mlevental/dev_projects/torch-mlir/hls/scripts/BraggNN.affine.mlir.dirty.llvm.unrollparfor.llvm.cse.cse.ll")
