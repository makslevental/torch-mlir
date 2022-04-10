set -e
torch-mlir-opt forward.mlir -torch-hls-promote-allocs -cse -symbol-dce -o forward.mlir.promoted
scalehls-translate forward.mlir.promoted --emit-hlspy --mlir-print-elementsattrs-with-hex-if-larger=-1 -o forward.py
black forward.py
python mlir_ops.py
python forward_rewritten.py > forward.ll