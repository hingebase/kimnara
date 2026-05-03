#include <iostream>
#if defined __cpp_lib_syncbuf && __cpp_lib_syncbuf >= 201803L
#  include <syncstream>
#endif

#include <nanobind/nanobind.h>

#ifdef _WIN32
constexpr unsigned RTLD_NOW = 2;
constexpr unsigned RTLD_NOLOAD = 4;

extern "C" {
__declspec(dllimport) void *dlopen(const char *, int);
__declspec(dllimport) void *dlsym(void *, const char *);
}
#else  // !_WIN32
#  include <dlfcn.h>
#endif  // _WIN32

#define KN_C_API Py_LOCAL_SYMBOL
#include "../include/kimnara.h"

namespace {
void KimnaraLogErrorHandler(
    [[maybe_unused]] void *logger,
    [[maybe_unused]] const char *msg,
    [[maybe_unused]] size_t size
) {
#if defined __cpp_lib_syncbuf && __cpp_lib_syncbuf >= 201803L
    std::osyncstream(std::cerr)
#else
    std::cerr
#endif
        << "Kimnara logging module hasn't been loaded\n";
}

using KimnaraLog = decltype(&KimnaraLogErrorHandler);

struct KimnaraProxy {
    KimnaraLog trace;
    KimnaraLog debug;
    KimnaraLog info;
    KimnaraLog warning;
    KimnaraLog error;
    KimnaraLog critical;
};

void *KimnaraOpen(PyObject *sys_modules, const char *name) {
    if (PyObject *module = PyDict_GetItemString(sys_modules, name)) [[likely]] {
        _Py_COMP_DIAG_PUSH
        _Py_COMP_DIAG_IGNORE_DEPR_DECLS
        ;
        return dlopen(
            // Expect success since Unix platforms already use UTF-8 encoding
            PyModule_GetFilename(module),
            // The mode argument must contain one of RTLD_LAZY/RTLD_NOW
            // according to the documentation. Python sys.getdlopenflags()
            // returns RTLD_NOW by default.
            RTLD_NOLOAD | RTLD_NOW
        );
        _Py_COMP_DIAG_POP
    }
    return nullptr;
}

const KimnaraProxy *KimnaraInit() {
    static const auto proxy = [] {
        nanobind::gil_scoped_acquire gil;
        PyObject *sys_modules = PyImport_GetModuleDict();
        if (
            void *module = KimnaraOpen(sys_modules, "kimnara.logging._spdlog")
        ) [[likely]] {
            // No need to call dlclose, see glibc/elf/tst-noload.c
            return KimnaraProxy {
                .trace = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraTrace")
                ),
                .debug = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraDebug")
                ),
                .info = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraInfo")
                ),
                .warning = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraWarning")
                ),
                .error = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraError")
                ),
                .critical = reinterpret_cast<KimnaraLog>(
                    dlsym(module, "KimnaraCritical")
                ),
            };
        }
        return KimnaraProxy {
            .trace = KimnaraLogErrorHandler,
            .debug = KimnaraLogErrorHandler,
            .info = KimnaraLogErrorHandler,
            .warning = KimnaraLogErrorHandler,
            .error = KimnaraLogErrorHandler,
            .critical = KimnaraLogErrorHandler,
        };
    }();
    return &proxy;
}
}  // namespace

extern "C" {
KN_C_API void KimnaraTrace(void *logger, const char *msg, size_t size) {
    KimnaraInit()->trace(logger, msg, size);
}

KN_C_API void KimnaraDebug(void *logger, const char *msg, size_t size) {
    KimnaraInit()->debug(logger, msg, size);
}

KN_C_API void KimnaraInfo(void *logger, const char *msg, size_t size) {
    KimnaraInit()->info(logger, msg, size);
}

KN_C_API void KimnaraWarning(void *logger, const char *msg, size_t size) {
    KimnaraInit()->warning(logger, msg, size);
}

KN_C_API void KimnaraError(void *logger, const char *msg, size_t size) {
    KimnaraInit()->error(logger, msg, size);
}

KN_C_API void KimnaraCritical(void *logger, const char *msg, size_t size) {
    KimnaraInit()->critical(logger, msg, size);
}
}
