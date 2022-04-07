scalehls-translate forward.mlir --emit-hlspy --mlir-print-elementsattrs-with-hex-if-larger=-1 -o forward.py
black forward.py
python mlir_ops.py
python forward_rewritten.py > forward.ll