if(UNIX)
    add_library(kimnara STATIC libkimnara.cpp)

    target_include_directories(
        kimnara
        PRIVATE
        ${Python_INCLUDE_DIRS}
        "${NB_DIR}/include"
    )

    install(
        TARGETS kimnara
        EXPORT kimnara_targets
        DESTINATION "${SKBUILD_PROJECT_NAME}/lib"
    )
else()
    add_library(kimnara INTERFACE)

    if(CMAKE_EXPORT_COMPILE_COMMANDS)
        add_library(phony OBJECT libkimnara.cpp)

        target_include_directories(
            phony
            PRIVATE
            ${Python_INCLUDE_DIRS}
            "${NB_DIR}/include"
        )
    endif()
endif()
