#pragma once

#if defined KN_INCLUDE_GUARD && !defined Py_LIMITED_API
#  ifdef Py_PYTHON_H
#    error Don't include Python.h before kimnara.h
#  endif  // Py_PYTHON_H
#  define Py_LIMITED_API 0x030a0000
#endif  // defined KN_INCLUDE_GUARD && !defined Py_LIMITED_API
#include <Python.h>

#ifndef KN_C_API
#  define KN_C_API Py_IMPORTED_SYMBOL
#endif  // KN_C_API

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

KN_C_API void KimnaraTrace(void *logger, const char *msg, size_t size);
KN_C_API void KimnaraDebug(void *logger, const char *msg, size_t size);
KN_C_API void KimnaraInfo(void *logger, const char *msg, size_t size);
KN_C_API void KimnaraWarning(void *logger, const char *msg, size_t size);
KN_C_API void KimnaraError(void *logger, const char *msg, size_t size);
KN_C_API void KimnaraCritical(void *logger, const char *msg, size_t size);

#if defined _WIN32 && defined SPDLOG_WCHAR_TO_UTF8_SUPPORT
KN_C_API void KimnaraTraceW(void *logger, const wchar_t *msg, size_t size);
KN_C_API void KimnaraDebugW(void *logger, const wchar_t *msg, size_t size);
KN_C_API void KimnaraInfoW(void *logger, const wchar_t *msg, size_t size);
KN_C_API void KimnaraWarningW(void *logger, const wchar_t *msg, size_t size);
KN_C_API void KimnaraErrorW(void *logger, const wchar_t *msg, size_t size);
KN_C_API void KimnaraCriticalW(void *logger, const wchar_t *msg, size_t size);
#endif  // defined _WIN32 && defined SPDLOG_WCHAR_TO_UTF8_SUPPORT

#ifdef __cplusplus
}
#endif  // __cplusplus
