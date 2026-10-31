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

__all__ = [
    "call_cfunc_in_numba",
    "erf",
    "erf_cydecl",
    "erf_libm_numba",
    "erf_scipy_ctypes",
    "erf_scipy_numba",
    "hello",
    "lstsq_ctypes",
    "lstsq_numba",
    "test_logging_capi",
]

import ctypes
import importlib.metadata
import math
import sys

import numpy as np
import optype.numpy as onp
from scipy.linalg import cython_lapack

import kimnara as kn


@kn.cfunc(nopython=True)
def _simple_cfunc() -> int:
    return 42


# This function would become uncacheable if `_simple_cfunc` was a
# `numba.cfunc`
@kn.func(nopython=True)
def call_cfunc_in_numba() -> int:
    return _simple_cfunc()


@kn.cdecl
def erf(x: np.float64) -> np.float64: ...


@kn.func(nopython=True)
def erf_libm_numba(x: float) -> float:
    return erf(x)


@kn.cydecl("scipy.special.cython_special", function_name="__pyx_fuse_1erf")
def erf_cydecl(x: np.float64, skip_dispatch: np.int32) -> np.float64: ...


def erf_scipy_ctypes(x: float) -> float:
    return erf_cydecl(x, 1)


@kn.func(nopython=True)
def erf_scipy_numba(x: np.float64) -> np.float64:
    return erf_cydecl(x, 1)


if sys.platform == "win32":
    @kn.func(nopython=True)
    def hello(
        name: onp.Array1D[np.uint8],
    ) -> tuple[onp.Array1D[np.uint8], onp.Array1D[np.uint8]]:
        buf1 = np.zeros(14, np.uint8)
        buf2 = np.zeros(16, np.uint8)
        fmt1 = np.array(
            (72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 0),
            np.uint8,
        )
        fmt2 = np.array(
            (72, 101, 108, 108, 111, 44, 32, 37, 115, 33, 0),
            np.uint8,
        )
        m: np.intc = kn.ccall(
            buf1.ctypes,
            fmt1.ctypes,
            lib="msvcrt",
            function_name="sprintf",
        )
        n: np.intc = kn.ccall(
            buf2.ctypes,
            fmt2.ctypes,
            name.ctypes,
            lib="msvcrt",
            function_name="sprintf",
        )
        if m != len(buf1) - 1 or n != len(buf2) - 1:
            empty = np.empty(0, np.uint8)
            return empty, empty
        return buf1, buf2
else:
    @kn.func(nopython=True)
    def hello(
        name: onp.Array1D[np.uint8],
    ) -> tuple[onp.Array1D[np.uint8], onp.Array1D[np.uint8]]:
        buf1 = np.zeros(14, np.uint8)
        buf2 = np.zeros(16, np.uint8)
        fmt1 = np.array(
            (72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33, 0),
            np.uint8,
        )
        fmt2 = np.array(
            (72, 101, 108, 108, 111, 44, 32, 37, 115, 33, 0),
            np.uint8,
        )
        m: np.intc = kn.ccall(
            buf1.ctypes,
            fmt1.ctypes,
            function_name="sprintf",
        )
        n: np.intc = kn.ccall(
            buf2.ctypes,
            fmt2.ctypes,
            name.ctypes,
            function_name="sprintf",
        )
        if m != len(buf1) - 1 or n != len(buf2) - 1:
            empty = np.empty(0, np.uint8)
            return empty, empty
        return buf1, buf2


@kn.gufunc("(n,m),(m)->(n)", align="mkl")
def lstsq_ctypes(
    a: onp.Array2D[kn.Mut[np.float64]],
    b: onp.Array1D[kn.Mut[np.float64]],
) -> onp.Array1D[np.float64]:
    # Using the syntax compatible with both ctypes and Numba
    # Use `byref` or `insert_unique_const` if no such concern
    n0, m0 = a.shape
    m = np.array((m0,), np.int32)
    n = np.array((n0,), np.int32)
    nrhs = np.array((1,), np.int32)
    lda = np.array(((m0 - 1 | 7) + 1,), np.int32)  # 64-byte padding
    rcond = np.array((-1.,), np.float64)
    rank = np.empty(1, np.int32)
    work = np.empty(1)
    lwork = np.array((-1,), np.int32)
    iwork = np.empty(1, np.int32)
    info = np.empty(1, np.int32)
    kn.cycall(
        m.ctypes,
        n.ctypes,
        nrhs.ctypes,
        None,  # a
        lda.ctypes,
        None,  # b
        lda.ctypes,  # ldb
        None,  # s
        rcond.ctypes,
        rank.ctypes,
        work.ctypes,
        lwork.ctypes,
        iwork.ctypes,
        info.ctypes,
        module=cython_lapack,
        function_name="dgelsd",
    )
    if info.item():
        kn.ops.invalid()
    s = np.empty(n0)
    work = np.empty(np.intp(work.item()))
    lwork[0] = len(work)
    iwork = np.empty(iwork.item(), np.int32)
    kn.cycall(
        m.ctypes,
        n.ctypes,
        nrhs.ctypes,
        a.ctypes,
        lda.ctypes,
        b.ctypes,
        lda.ctypes,  # ldb
        s.ctypes,
        rcond.ctypes,
        rank.ctypes,
        work.ctypes,
        lwork.ctypes,
        iwork.ctypes,
        info.ctypes,
        module=cython_lapack,
        function_name="dgelsd",
    )
    if info.item():
        kn.ops.invalid()
    return b[:2]  # pyright: ignore[reportReturnType]


@kn.gufunc("(n,m),(m)->(n)", align="mkl", nopython=True)
def lstsq_numba(
    a: onp.Array2D[kn.Mut[np.float64]],
    b: onp.Array1D[kn.Mut[np.float64]],
    x: onp.Array1D[np.float64],
) -> None:
    n0, m0 = a.shape
    m = np.array((m0,), np.int32)
    n = np.array((n0,), np.int32)
    nrhs = np.array((1,), np.int32)
    lda = np.array(((m0 - 1 | 7) + 1,), np.int32)
    rcond = np.array((-1.,), np.float64)
    rank = np.empty(1, np.int32)
    work = np.empty(1)
    lwork = np.array((-1,), np.int32)
    iwork = np.empty(1, np.int32)
    info = np.empty(1, np.int32)
    kn.cycall(
        m.ctypes,
        n.ctypes,
        nrhs.ctypes,
        None,  # a
        lda.ctypes,
        None,  # b
        lda.ctypes,  # ldb
        None,  # s
        rcond.ctypes,
        rank.ctypes,
        work.ctypes,
        lwork.ctypes,
        iwork.ctypes,
        info.ctypes,
        module=cython_lapack,
        function_name="dgelsd",
    )
    if info.item():
        x[:] = math.nan
        kn.ops.invalid()
        return
    s = np.empty(n0)
    work = np.empty(np.intp(work.item()))
    lwork[0] = len(work)
    iwork = np.empty(iwork.item(), np.int32)
    kn.cycall(
        m.ctypes,
        n.ctypes,
        nrhs.ctypes,
        a.ctypes,
        lda.ctypes,
        b.ctypes,
        lda.ctypes,  # ldb
        s.ctypes,
        rcond.ctypes,
        rank.ctypes,
        work.ctypes,
        lwork.ctypes,
        iwork.ctypes,
        info.ctypes,
        module=cython_lapack,
        function_name="dgelsd",
    )
    if info.item():
        x[:] = math.nan
        kn.ops.invalid()
        return
    x[:] = b[:2]


def test_logging_capi(logger: object) -> None:
    dll = ctypes.cdll[_locate_dll()]
    func = ctypes.CFUNCTYPE(None, ctypes.c_size_t)(("TestLoggingCAPI", dll))
    if not isinstance(logger, int):
        logger = kn.logging.get_pointer(logger)  # pyright: ignore[reportUnknownMemberType]
    func(logger)


def _locate_dll() -> str:
    match sys.platform:
        case "darwin":
            name = "libtest.dylib"
        case "win32":
            name = "test.dll"
        case p if p.startswith("linux"):
            name = "libtest.so"
        case _:
            raise NotImplementedError
    if paths := importlib.metadata.files("test-package-cmake"):
        for p in paths:
            if p.name == name:
                return str(p.locate())
    raise FileNotFoundError(name)
