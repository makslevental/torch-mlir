bash ./make_matmul.sh
bash ./torchmlir.sh
cp vitis_stuff/proj/solution1/impl/verilog/forward.v example/forward.v
mv matmul.llvm.mlir example/matmul.llvm.mlir