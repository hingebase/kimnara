#include <string_view>

#include <nanobind/nanobind.h>
#include <spdlog/spdlog.h>

#define KN_C_API Py_EXPORTED_SYMBOL
#include "../include/kimnara.h"

namespace {
template<spdlog::level::level_enum kLevel, typename T>
void Log(void *logger, const T *msg, size_t size) {
    auto *obj = reinterpret_cast<spdlog::logger *>(logger);
    if (size) {
        obj->log(kLevel, std::basic_string_view(msg, size));
    } else {
        obj->log(kLevel, msg);
    }
}
}  // namespace

extern "C" {
KN_C_API void KimnaraTrace(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::trace>(logger, msg, size);
}

KN_C_API void KimnaraDebug(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::debug>(logger, msg, size);
}

KN_C_API void KimnaraInfo(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::info>(logger, msg, size);
}

KN_C_API void KimnaraWarning(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::warn>(logger, msg, size);
}

KN_C_API void KimnaraError(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::err>(logger, msg, size);
}

KN_C_API void KimnaraCritical(void *logger, const char *msg, size_t size) {
    Log<spdlog::level::critical>(logger, msg, size);
}

#if defined _WIN32 && defined SPDLOG_WCHAR_TO_UTF8_SUPPORT
KN_C_API void KimnaraTraceW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::trace>(logger, msg, size);
}

KN_C_API void KimnaraDebugW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::debug>(logger, msg, size);
}

KN_C_API void KimnaraInfoW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::info>(logger, msg, size);
}

KN_C_API void KimnaraWarningW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::warn>(logger, msg, size);
}

KN_C_API void KimnaraErrorW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::err>(logger, msg, size);
}

KN_C_API void KimnaraCriticalW(void *logger, const wchar_t *msg, size_t size) {
    Log<spdlog::level::critical>(logger, msg, size);
}
#endif  // defined _WIN32 && defined SPDLOG_WCHAR_TO_UTF8_SUPPORT
}

// NOLINTBEGIN
NB_MODULE(_spdlog, m)
// NOLINTEND
{
    m.def("get_pointer", [](const nanobind::handle &logger) {
        return reinterpret_cast<uintptr_t>(
            nanobind::detail::nb_inst_ptr(logger.ptr())
        );
    });
}
