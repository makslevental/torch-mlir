include(AddMLIRPython)

declare_mlir_python_sources(HLSExtensions)

declare_mlir_python_extension(HLSExtensions.Main
        MODULE_NAME hls_utils
        ADD_TO_PARENT HLSExtensions
        SOURCES
        get_val_ident.cpp
        EMBED_CAPI_LINK_LIBS
        TorchMLIRCAPI
        PRIVATE_LINK_LIBS
        LLVMSupport
        )


set(_source_components
        HLSExtensions
        )

add_mlir_python_modules(HLSPythonModules
        ROOT_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/hls"
        INSTALL_PREFIX "hls"
        DECLARED_SOURCES ${_source_components}
        COMMON_CAPI_LINK_LIBS
        TorchMLIRAggregateCAPI
        )
