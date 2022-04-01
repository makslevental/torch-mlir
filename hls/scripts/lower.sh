set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/cmake-build-debug/bin"
ROOT_LIBDIR="${ROOT_DIR}/cmake-build-debug/lib"

FN=$1

# promote allocs
$ROOT_BINDIR/torch-mlir-opt $FN -torch-hls-promote-allocs -o $FN.promoted

# unroll reduction loops
$ROOT_BINDIR/torch-mlir-opt $FN.promoted -dirty-pass="unrollparfor=0 csts=1 unrollloops=1" -o $FN.dirty

# lower partially so you can unroll parloops
$ROOT_BINDIR/mlir-opt $FN.dirty \
  -convert-memref-to-llvm \
  -loop-invariant-code-motion \
  -o $FN.dirty.llvm

# unroll parallel loops
$ROOT_BINDIR/torch-mlir-opt $FN.dirty.llvm -dirty-pass="unrollparfor=1 csts=1 unrollloops=0" -o $FN.dirty.llvm.unrollparfor

# lower to llvm dialect
$ROOT_BINDIR/mlir-opt $FN.dirty.llvm.unrollparfor \
  -convert-memref-to-llvm \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
  -reconcile-unrealized-casts -o $FN.dirty.llvm.unrollparfor.llvm

# cse llvm
#$ROOT_BINDIR/torch-mlir-opt $FN.dirty.llvm.unrollparfor.llvm -dirty-pass-llvm="unrollparfor=0" -strip-debuginfo -o $FN.dirty.llvm.unrollparfor.llvm.cse
python read_large_file.py $FN.dirty.llvm.unrollparfor.llvm

# translate to llvm ir
$ROOT_BINDIR/mlir-translate $FN.dirty.llvm.unrollparfor.llvm.cse.cse \
  -mlir-to-llvmir -o $FN.dirty.llvm.unrollparfor.llvm.cse.cse.ll

# llvm ir opts
#  -instcombine -instcombine-max-iterations=10 \
${ROOT_BINDIR}/opt -debug $FN.dirty.llvm.unrollparfor.llvm.cse.cse.ll \
  -S \
  -enable-new-pm=0 \
  -load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
  -strip-attr \
  -time-passes \
  -stats \
  -mem2arr \
  -strip-debug \
  -xlnmath \
  -xlnname \
  -o $FN.dirty.llvm.unrollparfor.llvm.cse.cse.ll.vitis

#
##sed -i 's/immarg//g' $dst_file
##sed -i 's/noundef//g' $dst_file
##
##${ROOT_BINDIR}/opt "${dst_file}" > test.ll
##
###cp wrapper.cpp vitis_stuff/wrapper.cpp
###cp run_hls.tcl vitis_stuff/run_hls.tcl
#
##  -xlnunroll \
##  -xlnarraypartition \
