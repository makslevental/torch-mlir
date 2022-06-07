set(LLVM_TARGET_DEFINITIONS BraggHLSPasses.td)
mlir_tablegen(BraggHLSPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(BraggHLSIncGen)

add_mlir_library(BraggHLS
        BraggHLSDropPublicReturn.cpp
        BraggHLSPromoteAllocs.cpp
        BraggHLSPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/hls/mlir

        DEPENDS
        BraggHLSIncGen
        MLIRLinalgPassIncGen
        )

target_link_options(BraggHLS PUBLIC -undefined dynamic_lookup)
torch_mlir_target_includes(BraggHLS)