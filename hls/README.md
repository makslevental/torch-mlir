```shell
cmake -GNinja -Bbuild \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DPython3_EXECUTABLE=/home/mlevental/dev_projects/torch-mlir/mlir_venv/bin/python \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=torch-mlir \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR=/home/mlevental/dev_projects/torch-mlir \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DMLIR_ENABLE_BINDINGS_PYTHON=true \
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=/usr/local/bin/mold" \
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=/usr/local/bin/mold" \
    external/llvm-project/llvm
```

```shell
/home/mlevental/dev_projects/torch-mlir/build_tools/update_torch_ods.sh

/home/mlevental/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/212.5457.51/bin/cmake/linux/bin/cmake --build /home/mlevental/dev_projects/torch-mlir/build --target MLIRTorchOpsIncGen
/home/mlevental/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/212.5457.51/bin/cmake/linux/bin/cmake --build /home/mlevental/dev_projects/torch-mlir/build --target all
/home/mlevental/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/212.5457.51/bin/cmake/linux/bin/cmake --build /home/mlevental/dev_projects/torch-mlir/build --target torch-mlir-opt

../../build/bin/torch-mlir-opt -pass-pipeline='tensor-constant-bufferize,builtin.func(scf-bufferize),builtin.func(linalg-bufferize),builtin.func(std-bufferize),builtin.func(tensor-bufferize),func-bufferize,builtin.func(finalizing-bufferize)' matmul.llvm.mlir > matmul.2.llvm.mlir

../../build/bin/torch-mlir-opt -pass-pipeline='builtin.func(convert-linalg-to-loops),builtin.func(lower-affine),builtin.func(convert-scf-to-std),builtin.func(refback-expand-ops-for-llvm),builtin.func(arith-expand),builtin.func(convert-math-to-llvm)' matmul.2.llvm.mlir > matmul.3.llvm.mlir

../../build/bin/torch-mlir-opt -pass-pipeline='convert-memref-to-llvm,convert-std-to-llvm,reconcile-unrealized-casts' matmul.3.llvm.mlir > matmul.4.llvm.mlir

cmake --build build --target all

```