set(OpenMP_RUNTIME_MSVC llvm)
find_package(OpenMP REQUIRED)

find_package(TBB 2021.6.0...<2024.0.0 REQUIRED CONFIG)

kimnara_add_module(_openmp openmp.cpp)
kimnara_add_module(_tbb tbb.cpp)

target_link_libraries(_tbb PRIVATE TBB::tbb)

# PyPI package `intel-openmp ==2024.1.0`
if(NOT APPLE AND NOT DEFINED ENV{CONDA_BUILD})
    target_include_directories(
        _openmp
        PRIVATE
        "${PYTHON_SYS_PREFIX}/opt/compiler/include"
    )
endif()

if(
    LINUX OR (
        APPLE
        AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64"
        AND NOT DEFINED ENV{CONDA_BUILD}
    )
)
    # https://discourse.llvm.org/t/libiomp-not-libgomp-as-default-library-linked-with-fopenmp/36444/20
    # https://stackoverflow.com/a/25988011
    target_compile_options(_openmp PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(_openmp PRIVATE iomp5)
    if(APPLE OR NOT DEFINED ENV{CONDA_BUILD})
        target_link_directories(_openmp PRIVATE "${PYTHON_SYS_PREFIX}/lib")
    endif()
else()
    target_link_libraries(_openmp PRIVATE OpenMP::OpenMP_CXX)
    if(MSVC)
        target_link_options(_openmp PRIVATE "/nodefaultlib:libomp")
        target_link_libraries(_openmp PRIVATE libiomp5md)
        if(NOT DEFINED ENV{CONDA_BUILD})
            target_include_directories(
                _openmp
                PRIVATE
                "${PYTHON_SYS_PREFIX}/Library/include"
            )
            target_link_directories(
                _openmp
                PRIVATE
                "${PYTHON_SYS_PREFIX}/Library/lib"
            )
        endif()
    endif()
endif()

# https://github.com/pypa/auditwheel/issues/552
# https://scikit-build-core.readthedocs.io/en/latest/guide/dynamic_link.html
if(UNIX AND NOT DEFINED ENV{CONDA_BUILD})
    if(APPLE)
        set(ORIGIN_TOKEN "@loader_path")
    else()
        set(ORIGIN_TOKEN "$ORIGIN")
    endif()
    set_property(
        TARGET _openmp _tbb
        PROPERTY INSTALL_RPATH "${ORIGIN_TOKEN}/../../../.."
    )
endif()

install(TARGETS _openmp _tbb DESTINATION "${SKBUILD_PROJECT_NAME}/threading")
