# Copyright 2026 hingebase

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Test kimnara.logging API."""

import ctypes
import gc
import pathlib
from collections.abc import Callable
from typing import Protocol, no_type_check

import spydlog as spd  # pyright: ignore[reportMissingTypeStubs]
import test_package_cmake

import kimnara as kn

_CAPI = Callable[[int, bytes | str, int], None]


def test_logging_capi(tmp_path: pathlib.Path) -> None:
    """Test logging configuration in Python and logging in C."""
    (
        trace, debug, info, warn, error, critical,
        trace_w, debug_w, info_w, warn_w, error_w, critical_w,
    ) = _get_logging_capi()

    log_file = tmp_path / "test.log"
    logger, ptr = _setup_logger(log_file)
    test_package_cmake.test_logging_capi(ptr)

    trace(ptr, b"Trace message", 0)

    message = b"Debug message"
    debug(ptr, message, len(message))

    message = b"Info message"
    info(ptr, message, len(message) - 1)

    warn(ptr, b"Warning message", 0)

    message = b"Error message"
    error(ptr, message, len(message))

    message = b"Critical message"
    critical(ptr, message, len(message) - 1)

    if trace_w:
        assert debug_w
        assert info_w
        assert warn_w
        assert error_w
        assert critical_w

        trace_w(ptr, "Trace message", 0)

        message = "Debug message"
        debug_w(ptr, message, len(message))

        message = "Info message"
        info_w(ptr, message, len(message) - 1)

        warn_w(ptr, "Warning message", 0)

        message = "Error message"
        error_w(ptr, message, len(message))

        message = "Critical message"
        critical_w(ptr, message, len(message) - 1)
    else:
        assert not debug_w
        assert not info_w
        assert not warn_w
        assert not error_w
        assert not critical_w
    logger.flush()
    del logger
    gc.collect()
    _check_log_file_content(log_file, bool(trace_w) + 2)


class _Logger(Protocol):
    def flush(self) -> None: ...


def _check_log_file_content(log_file: pathlib.Path, n: int) -> None:
    with log_file.open(encoding="ascii") as f:
        for _ in range(n):
            assert next(f) == "warning  Warning message\n"
            assert next(f) == "error    Error message\n"
            # Truncated intentionally
            assert next(f) == "critical Critical messag\n"
        assert not next(f, "")


def _get_logging_capi() -> tuple[
    _CAPI,
    _CAPI,
    _CAPI,
    _CAPI,
    _CAPI,
    _CAPI,
    _CAPI | None,
    _CAPI | None,
    _CAPI | None,
    _CAPI | None,
    _CAPI | None,
    _CAPI | None,
]:
    dll = ctypes.cdll[kn.logging.get_library()]

    def capi(name: str) -> _CAPI | None:
        c_function_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_size_t,
            ctypes.c_wchar_p if name.endswith("W") else ctypes.c_char_p,
            ctypes.c_size_t,
        )
        try:
            return c_function_type((name, dll))
        except AttributeError:
            return None

    trace = capi("KimnaraTrace")
    debug = capi("KimnaraDebug")
    info = capi("KimnaraInfo")
    warn = capi("KimnaraWarning")
    error = capi("KimnaraError")
    critical = capi("KimnaraCritical")
    trace_w = capi("KimnaraTraceW")
    debug_w = capi("KimnaraDebugW")
    info_w = capi("KimnaraInfoW")
    warn_w = capi("KimnaraWarningW")
    error_w = capi("KimnaraErrorW")
    critical_w = capi("KimnaraCriticalW")
    assert trace
    assert debug
    assert info
    assert warn
    assert error
    assert critical
    return (
        trace, debug, info, warn, error, critical,
        trace_w, debug_w, info_w, warn_w, error_w, critical_w,
    )


@no_type_check
def _setup_logger(log_file: pathlib.Path) -> tuple[_Logger, int]:
    logger = spd.basic_logger_st("kimnara.test", str(log_file), truncate=True)
    logger.set_level(spd.level.warn)
    logger.set_pattern("%-8l %v")
    return logger, kn.logging.get_pointer(logger)
