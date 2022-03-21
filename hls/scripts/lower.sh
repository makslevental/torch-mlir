DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/cmake-build-debug/bin"
ROOT_LIBDIR="${ROOT_DIR}/cmake-build-debug/lib"

FN=$1
$ROOT_BINDIR/mlir-opt \
  -promote-buffers-to-stack \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-memref-to-llvm \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' \
  -reconcile-unrealized-casts < $FN > $FN.llvm

$ROOT_BINDIR/mlir-translate \
  -mlir-to-llvmir < $FN.llvm > $FN.ll

${ROOT_BINDIR}/opt $FN.ll \
  -S \
  -enable-new-pm=0 \
  -load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
  -strip-attr \
  -memcpyopt \
  -mem2arr \
  -memcpyopt \
  -strip-debug \
  -instcombine \
  -xlnmath \
  -xlnname \
  -verify \
  > braggnn.opt.vitis.ll
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
