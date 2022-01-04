#set(LLVM_TARGET_DEFINITIONS HLSTorchOps.td)
#mlir_tablegen(HLSTorchOps.h.inc -gen-op-decls)
#mlir_tablegen(HLSTorchOps.cpp.inc -gen-op-defs)
#mlir_tablegen(HLSTorchDialect.h.inc -gen-dialect-decls -dialect=torch)
#mlir_tablegen(HLSTorchDialect.cpp.inc -gen-dialect-defs -dialect=torch)
#add_public_tablegen_target(MLIRHLSTorchOpsIncGen)
#add_dependencies(mlir-headers MLIRHLSTorchOpsIncGen)

#set(LLVM_TARGET_DEFINITIONS HLSTorchTypes.td)
#mlir_tablegen(HLSTorchTypes.h.inc -gen-typedef-decls)
#mlir_tablegen(HLSTorchTypes.cpp.inc -gen-typedef-defs)
#add_public_tablegen_target(MLIRHLSTorchTypesIncGen)

set(LLVM_TARGET_DEFINITIONS HLSPasses.td)
mlir_tablegen(HLSPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(HLSTorchIncGen)

#        HLSConvertOperators.cpp
#        HLSTorchOps.cpp
#        HLSAdjustCallingConventions.cpp
#        HLSReduceOpVariants.cpp
add_mlir_library(HLSTorch
        HLSDecomposeComplexOps.cpp
        HLSDropPublicReturn.cpp
        HLSPromoteAllocs.cpp
        HLSLinalgBufferize.cpp
        HLSRefineTypes.cpp
        HLSTorchToLinalg.cpp
        HLSPasses.cpp
        MemoryPlanning.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/hls/mlir

        DEPENDS
        HLSTorchIncGen
        MLIRLinalgPassIncGen
        )
#target_link_options(HLSTorch PUBLIC "-Wl,--unresolved-symbols=ignore-all")
target_link_options(HLSTorch PUBLIC -undefined dynamic_lookup)
torch_mlir_target_includes(HLSTorch)