include(AddMLIRPython)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)
set(HLS_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hls")

declare_mlir_python_sources(HLSPythonSources)
declare_mlir_python_sources(HLSPythonExtensions)
declare_mlir_python_extension(HLSPythonExtensions.Main
        MODULE_NAME hls_utils
        ADD_TO_PARENT HLSPythonExtensions
        SOURCES
        get_val_ident.cpp
        EMBED_CAPI_LINK_LIBS
        HLSCAPI
        PRIVATE_LINK_LIBS
        LLVMSupport
        MLIRIR
        MLIRSupport
        )

set(_source_components
        MLIRPythonSources
        HLSPythonSources
        HLSPythonExtensions
        )

if (NOT HLS_PYTHON_PACKAGES_DIR)
    set(HLS_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/python_packages")
endif ()

add_mlir_python_modules(HLSPythonModules
        ROOT_PREFIX "${HLS_PYTHON_PACKAGES_DIR}/hls/hls"
        INSTALL_PREFIX "python_packages/hls/hls"
        DECLARED_SOURCES ${_source_components}
        #        COMMON_CAPI_LINK_LIBS
        #        HLSAggregateCAPI
        )