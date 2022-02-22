DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/cmake-build-debug/bin"
ROOT_LIBDIR="${ROOT_DIR}/cmake-build-debug/lib"

FN=$1
$ROOT_BINDIR/mlir-opt \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-memref-to-llvm \
  -convert-arith-to-llvm \
  -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' \
  -reconcile-unrealized-casts < /home/mlevental/dev_projects/torch-mlir/hls/scripts/$FN.affine.mlir > /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/$FN.llvm.mlir

$ROOT_BINDIR/mlir-translate \
  -mlir-to-llvmir < /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/$FN.llvm.mlir > /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/$FN.ll

src_file=/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/$FN.ll
dst_file=/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/$FN.opt.vitis.ll
"${ROOT_BINDIR}/opt" "${src_file}" \
  -S \
  -enable-new-pm=0 \
  -load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
  -mem2arr \
  -strip-debug \
  -instcombine \
  -xlnmath \
  -xlnname \
  -strip-attr \
  -xlnunroll \
  -xlnarraypartition \
  > "${dst_file}"
#
##cp wrapper.cpp vitis_stuff/wrapper.cpp
##cp run_hls.tcl vitis_stuff/run_hls.tcl
