#!/bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/cmake-build-debug/bin"
ROOT_LIBDIR="${ROOT_DIR}/cmake-build-debug/lib"

FN=$1

$ROOT_BINDIR/mlir-opt -debug \
  $FN.affine.baseline.mlir \
  -affine-loop-invariant-code-motion \
  -affine-simplify-structures \
  -affine-loop-invariant-code-motion \
  -affine-loop-normalize \
  -affine-loop-invariant-code-motion \
  -affine-loop-fusion="fusion-maximal=true mode=greedy" \
  -affine-loop-invariant-code-motion \
  -affine-simplify-structures \
  -affine-loop-invariant-code-motion \
  -affine-loop-normalize \
  -affine-loop-invariant-code-motion \
  -affine-loop-unroll="unroll-full=true" > $FN.affine.mlir

$ROOT_BINDIR/mlir-opt \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-memref-to-llvm \
  -convert-arith-to-llvm \
  -convert-std-to-llvm='use-bare-ptr-memref-call-conv=1' \
  -reconcile-unrealized-casts <$FN.affine.mlir > vitis_stuff/$FN.llvm.mlir

$ROOT_BINDIR/mlir-translate \
  -mlir-to-llvmir <vitis_stuff/$FN.llvm.mlir > vitis_stuff/$FN.ll

src_file=vitis_stuff/$FN.ll
dst_file=vitis_stuff/$FN.opt.vitis.ll
${ROOT_BINDIR}/opt "${src_file}" \
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
  > "${dst_file}"
