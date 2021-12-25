set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(HLSTransformsIncGen)

add_mlir_library(HLSTransforms
        HLSDecomposeComplexOps.cpp
        HLSDropPublicReturn.cpp
        HLSLinalgBufferize.cpp
        HLSRefineTypes.cpp
        HLSTorchToLinalg.cpp
        Passes.cpp

        DEPENDS
        HLSTransformsIncGen
        MLIRLinalgPassIncGen
        )
target_link_options(HLSTransforms PUBLIC "-Wl,--unresolved-symbols=ignore-all")
#target_link_options(HLSTransforms PUBLIC -undefined dynamic_lookup)
torch_mlir_target_includes(HLSTransforms)