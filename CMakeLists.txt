cmake_minimum_required(VERSION 4.0)

if(APPLE)
    if(DEFINED ENV{CONDA_BUILD})
        # CMake requires a prefix for clang-scan-deps while conda-forge doesn't
        # provide it, see https://gitlab.kitware.com/cmake/cmake/-/blob/v4.0.0/Modules/Compiler/Clang-FindBinUtils.cmake#L51-54
        set(
            CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
            "$ENV{BUILD_PREFIX}/bin/clang-scan-deps"
        )
    elseif(
        "$ENV{PIXI_PROJECT_NAME}" STREQUAL "kimnara"
        AND "$ENV{PIXI_ENVIRONMENT_NAME}" STREQUAL "clangd"
    )
        set(CMAKE_CXX_COMPILER_AR "$ENV{CONDA_PREFIX}/bin/llvm-ar")
        set(
            CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
            "$ENV{CONDA_PREFIX}/bin/clang-scan-deps"
        )
        set(CMAKE_CXX_COMPILER_RANLIB "$ENV{CONDA_PREFIX}/bin/llvm-ranlib")
    else()
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()

if(DEFINED SKBUILD_CORE_VERSION)
    project(
        ${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX
    )
else()
    # Allow invoking CMake directly, for the purpose of debugging
    set(SKBUILD_PROJECT_NAME kimnara)
    project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

set(CMAKE_CXX_STANDARD 20)

include(CheckIPOSupported)
check_ipo_supported()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(
    Python 3.10.0...<3.15.0
    REQUIRED Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
)

if(DEFINED SKBUILD_CORE_VERSION)
    find_package(nanobind 2.10.2...<3.0.0 REQUIRED CONFIG)
else()
    find_package(
        nanobind 2.10.2...<3.0.0
        REQUIRED
        CONFIG
        PATHS "${Python_SITELIB}/nanobind"
    )
endif()

function(kimnara_add_module NAME SOURCE)
    nanobind_add_module(
        ${NAME}
        STABLE_ABI  # Will be ignored on unsupported Python versions
        LTO
        NB_DOMAIN ${SKBUILD_PROJECT_NAME}
        ${SOURCE}
    )
endfunction()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sys;print(end=sys.prefix)"
    OUTPUT_VARIABLE PYTHON_SYS_PREFIX
    ERROR_QUIET
    COMMAND_ERROR_IS_FATAL ANY
)

if(WIN32 AND NOT DEFINED ENV{CONDA_BUILD})
    list(APPEND CMAKE_PREFIX_PATH "${PYTHON_SYS_PREFIX}/Library")
endif()

# For dependencies available on conda-forge but not PyPI
# Prefer the conda-forge version if available
include(FetchContent)

add_subdirectory(src/kimnara/align)
add_subdirectory(src/kimnara/lib)
add_subdirectory(src/kimnara/logging)
add_subdirectory(src/kimnara/threading)

if(WIN32)
    install(TARGETS kimnara EXPORT kimnara_targets)
endif()
install(
    EXPORT kimnara_targets
    DESTINATION "${SKBUILD_PROJECT_NAME}/cmake"
    NAMESPACE kimnara::
    FILE kimnara-targets.cmake
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/kimnara/cmake/kimnara-config.cmake.in
    kimnara-config.cmake
    INSTALL_DESTINATION "${SKBUILD_PROJECT_NAME}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
if(DEFINED SKBUILD_CORE_VERSION)
    write_basic_package_version_file(
        kimnara-config-version.cmake
        # TODO: Switch to SameMajorVersion once we reached 1.0.0
        COMPATIBILITY SameMinorVersion
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/kimnara-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/kimnara-config-version.cmake"
        DESTINATION "${SKBUILD_PROJECT_NAME}/cmake"
    )

    cmake_path(
        APPEND
            Python_SITELIB
            "nanobind-${nanobind_VERSION}.dist-info"
            licenses
            LICENSE
        OUTPUT_VARIABLE NANOBIND_LICENSE
    )
    install(
        FILES "${NANOBIND_LICENSE}"
        DESTINATION "${SKBUILD_METADATA_DIR}/licenses/nanobind"
    )

    if(SKBUILD_STATE STREQUAL editable)
        # Already included in sdist/wheel
        install(
            FILES src/kimnara/include/kimnara.h
            DESTINATION "${SKBUILD_PROJECT_NAME}/include"
        )
    endif()
else()
    add_library(
        phony2
        OBJECT
        packages/test-package-cmake/src/test_package_cmake/test.cpp
    )
    target_include_directories(
        phony2
        PRIVATE
        src/kimnara/include
        ${Python_INCLUDE_DIRS}
    )
endif()
