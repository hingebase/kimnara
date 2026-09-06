# Follow https://github.com/conda-forge/mimalloc-feedstock/blob/main/recipe/build.sh
# for compatibility
set(MI_OVERRIDE OFF)
set(MI_LOCAL_DYNAMIC_TLS ON)

set(MI_BUILD_SHARED OFF)
set(MI_BUILD_STATIC OFF)
set(MI_BUILD_TESTS OFF)

FetchContent_Declare(
    mimalloc
    GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
    GIT_TAG v3.5.1
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 3.1.5...<4.0.0
)
FetchContent_MakeAvailable(mimalloc)

kimnara_add_module(_mimalloc mimalloc.cpp)

if(DEFINED mimalloc_CONFIG)
    target_link_libraries(_mimalloc PRIVATE mimalloc)
else()
    target_link_libraries(_mimalloc PRIVATE mimalloc-obj)
    if(DEFINED SKBUILD_CORE_VERSION)
        install(
            FILES "${mimalloc_SOURCE_DIR}/LICENSE"
            DESTINATION "${SKBUILD_METADATA_DIR}/licenses/mimalloc"
        )
    endif()
endif()

install(TARGETS _mimalloc DESTINATION "${SKBUILD_PROJECT_NAME}/align")
