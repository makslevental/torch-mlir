LAYER=batchnorm2d
bash ./make_matmul.sh
bash ./torchmlir.sh
cp vitis_stuff/proj/solution1/impl/verilog/forward.v $LAYER/forward.v
mv $LAYER.llvm.mlir $LAYER/$LAYER.llvm.mlir