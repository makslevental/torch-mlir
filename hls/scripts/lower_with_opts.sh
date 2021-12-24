#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="${DIR}/../.."
ROOT_BINDIR="${ROOT_DIR}/cmake-build-debug/bin"
ROOT_LIBDIR="${ROOT_DIR}/cmake-build-debug/lib"

FN=$1
#  -affine-loop-fusion="fusion-maximal=true mode=greedy" \

time $ROOT_BINDIR/mlir-opt \
  $FN \
  -affine-loop-invariant-code-motion \
  -affine-simplify-structures \
  -affine-loop-normalize \
  -affine-loop-unroll="unroll-full=true" \
  -affine-loop-unroll="unroll-full=true" \
  -affine-loop-unroll="unroll-full=true" \
  -affine-loop-unroll="unroll-full=true" \
  -affine-loop-unroll="unroll-full=true" \
  -affine-loop-unroll="unroll-full=true" \
  > $FN.affine_opt7

time $ROOT_BINDIR/mlir-opt \
  -lower-affine \
  -convert-scf-to-cf \
  -convert-memref-to-llvm \
  -convert-math-to-llvm \
  -convert-arith-to-llvm \
  -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
  -reconcile-unrealized-casts \
  < $FN.affine_opt7 > $FN.llvm

time $ROOT_BINDIR/mlir-translate \
  -mlir-to-llvmir < $FN.llvm > $FN.ll

time ${ROOT_BINDIR}/opt $FN.ll \
  -S \
  -enable-new-pm=0 \
  -load "${ROOT_LIBDIR}/VhlsLLVMRewriter.so" \
  -strip-attr \
  -memcpyopt \
  -mem2arr \
  -strip-debug \
  -instcombine \
  -xlnmath \
  -xlnname \
  > braggnn.opt.vitis.ll
