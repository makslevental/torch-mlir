DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/build/bin"
ROOT_LIBDIR="${ROOT_DIR}/build/lib"

$ROOT_BINDIR/mlir-opt -lower-affine -convert-scf-to-std -convert-memref-to-llvm -convert-arith-to-llvm -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' -reconcile-unrealized-casts < /home/mlevental/dev_projects/torch-mlir/hls/scripts/braggnn.affine.mlir > /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.llvm.mlir
$ROOT_BINDIR/mlir-translate -mlir-to-llvmir < /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.llvm.mlir > /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.ll
#$ROOT_BINDIR/opt -enable-new-pm=0 -load /home/mlevental/dev_projects/torch-mlir/hls/scripts/../../build/lib/VhlsLLVMRewriter.so -mem2arr -instcombine -strip-debug -S < /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.ll > /home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.ll
src_file=/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.ll
dst_file=/home/mlevental/dev_projects/torch-mlir/hls/scripts/vitis_stuff/braggnn.opt.vitis.ll
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

cp wrapper.cpp vitis_stuff/wrapper.cpp