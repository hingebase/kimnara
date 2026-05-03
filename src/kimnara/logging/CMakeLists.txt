set(SPDLOG_PREVENT_CHILD_FD ON)
set(SPDLOG_NO_THREAD_ID ON)
set(SPDLOG_NO_ATOMIC_LEVELS ON)
set(SPDLOG_DISABLE_DEFAULT_LOGGER ON)
if(WIN32)
    set(SPDLOG_USE_STD_FORMAT ON)
    set(SPDLOG_WCHAR_SUPPORT ON)
    set(SPDLOG_WCHAR_FILENAMES ON)
    set(SPDLOG_WCHAR_CONSOLE ON)
endif()

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 1.16.0...<2.0.0
)
FetchContent_MakeAvailable(spdlog)

kimnara_add_module(_spdlog spdlog.cpp)

if(DEFINED spdlog_CONFIG)
    target_link_libraries(_spdlog PRIVATE spdlog::spdlog)

    # We have no chance to know whether spdlog was built with wchar support.
    # Assuming false here, but downstream packages may override it with
    # `#define SPDLOG_WCHAR_TO_UTF8_SUPPORT`.
else()
    target_link_libraries(_spdlog PRIVATE spdlog::spdlog_header_only)

    if(WIN32)
        target_compile_definitions(
            kimnara
            INTERFACE
            SPDLOG_WCHAR_TO_UTF8_SUPPORT
        )
    endif()

    if(DEFINED SKBUILD_CORE_VERSION)
        install(
            FILES "${spdlog_SOURCE_DIR}/LICENSE"
            DESTINATION "${SKBUILD_METADATA_DIR}/licenses/spdlog"
        )
    endif()
endif()

install(TARGETS _spdlog DESTINATION "${SKBUILD_PROJECT_NAME}/logging")
if(WIN32)
    install(
        FILES "$<TARGET_FILE_DIR:_spdlog>/_spdlog.lib"
        DESTINATION "${SKBUILD_PROJECT_NAME}/logging"
    )
endif()
