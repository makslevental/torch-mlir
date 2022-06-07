set -e
torch-mlir-opt forward.mlir -torch-hls-promote-allocs -cse -symbol-dce -o forward.mlir.promoted
python - <<EOF

import re

arg_pat = r"%([a-zA-Z0-9]*)"
const_pat = r"%c(\d+)"
patt1 = rf"scf\.for {arg_pat} = {const_pat} to {const_pat} step {const_pat}"
patt2 = rf"scf\.parallel \({arg_pat}, {arg_pat}, {arg_pat}, {arg_pat}\) = \({const_pat}, {const_pat}, {const_pat}, {const_pat}\) to \({const_pat}, {const_pat}, {const_pat}, {const_pat}\) step \({const_pat}, {const_pat}, {const_pat}, {const_pat}\)"

cst_map = {}

old_lines = open("forward.mlir.promoted").readlines()
new_lines = []
for i, line in enumerate(old_lines):
    if "arith.constant" in line and "index" in line:
        matches = re.findall(r"%c([\d|_]+) = arith.constant (\d+) : index", line)[0]
        assert len(matches) == 2
        cst_map[f"%c{matches[0]}"] = matches[1]
    if "scf.for" in line or "scf.parallel" in line:
        for cst_ident, cst in cst_map.items():
            line = line.replace(cst_ident, cst)
        line = line.replace("scf", "affine")

    new_lines.append(line)

open("forward.affine.mlir", "w").writelines(new_lines)

EOF
sed -i.bak 's/scf\.yield//g' forward.affine.mlir

bragghls-translate forward.affine.mlir --emit-hlspy --mlir-print-elementsattrs-with-hex-if-larger=-1 -o forward.py

python ../../python/interpreter/rewriters/python.py forward.py --macs
FN=macs python forward_rewritten.py
#
python ../../python/interpreter/rewriters/python.py forward.py
FN=regular python forward_rewritten.py

python ../../python/interpreter/cpp_scheduler.py "$PWD"/forward_regular.cpp

python ../../python/interpreter/rewriters/make_verilog.py "$PWD"/design.json
