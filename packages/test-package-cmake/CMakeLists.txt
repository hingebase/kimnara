cmake_minimum_required(VERSION 4.0)

if(APPLE)
    if(DEFINED ENV{CONDA_BUILD})
        set(
            CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS
            "$ENV{BUILD_PREFIX}/bin/clang-scan-deps"
        )
    else()
        set(CMAKE_LINKER_TYPE LLD)
    endif()
endif()

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

include(CheckIPOSupported)
check_ipo_supported()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(kimnara REQUIRED CONFIG)

add_library(test SHARED src/test_package_cmake/test.cpp)
target_link_libraries(test PRIVATE kimnara::kimnara)

install(
    TARGETS test
    DESTINATION "${SKBUILD_PROJECT_NAME}"
)
