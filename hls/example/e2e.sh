bash ./make_matmul.sh
bash ./torchmlir.sh
cp vitis_stuff/proj/solution1/impl/verilog/forward.v reprs/forward.v
mv matmul.llvm.mlir reprs/matmul.llvm.mlir