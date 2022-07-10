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

#sed -i.bak 's/memref\.load/affine.load/g' forward.affine.mlir
#sed -i.bak 's/memref\.store/affine.store/g' forward.affine.mlir
#torch-mlir-opt -affine-loop-unroll="unroll-full unroll-full-threshold=1000" -o forward.affine.unrolled.mlir
#torch-mlir-opt -debug -affine-scalrep forward.affine.unrolled.mlir -o forward.affine.unrolled.scalrep.mlir

scalehls-translate forward.affine.mlir --emit-hlspy --mlir-print-elementsattrs-with-hex-if-larger=-1 -o forward.py

python ../../scripts/mlir_ops.py forward.py
FN=regular python forward_rewritten_unrolled.py

circt-opt forward_regular.mlir -test-lp-scheduler=with=Problem -allow-unregistered-dialect -o forward_regular.sched.mlir
#/Users/mlevental/dev_projects/circt/build/clion/bin/circt-opt forward_regular.mlir -test-cpsat-scheduler=with=SharedOperatorsProblem -allow-unregistered-dialect -o forward_regular.sched.mlir

#python ../../scripts/mlir_val.py "$PWD"/forward_regular.sched.mlir
#
#python ../../scripts/make_verilog_mlir.py "$PWD"/design.json


#highlight_objects -color green -leaf_cells [get_cells _forward_inner/fadd*]
#highlight_objects -color red -leaf_cells [get_cells _forward_inner/fmul*]
#highlight_objects -color red -leaf_cells [get_cells sigProd*]
#highlight_objects -color red -leaf_cells [get_cells sigProd*]
#highlight_objects -color red -leaf_cells [get_cells sticky_d1*]
#highlight_objects -color red -leaf_cells [get_cells sign_d1*]
#highlight_objects -color red -leaf_cells [get_cells _forward_inner/sign_d1*]
#highlight_objects -color red -leaf_cells [get_cells exc_d1*]
#highlight_objects -color red -leaf_cells [get_cells expSig*]
#highlight_objects -color red -leaf_cells [get_cells expSumPre*]
#highlight_objects -color red -leaf_cells [get_cells fmul*]