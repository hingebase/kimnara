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

"""Test calling foreign functions."""

import math
import sys

import hypothesis
import hypothesis.strategies as st
import numpy as np
import optype.numpy as onp
import pytest
import scipy.special
import test_package_cmake
from numba.core import caching  # pyright: ignore[reportMissingTypeStubs]

import kimnara as kn

_PREFIX = "$_" if sys.platform == "darwin" else "$"


@hypothesis.given(st.floats())
def test_cdecl(x: float) -> None:
    """Test `kn.cdecl`."""
    actual = test_package_cmake.erf(x)
    desired = math.erf(x)
    # `actual == desired` fails on NaN
    np.testing.assert_equal(actual, desired)

    actual = test_package_cmake.erf_libm_numba(x)
    assert isinstance(actual, float)
    np.testing.assert_equal(actual, desired)

    dispatcher = test_package_cmake.erf_libm_numba.dispatcher
    # ruff: disable[private-member-access]
    cache = dispatcher._cache  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
    assert isinstance(cache, caching.FunctionCache)
    if dispatcher._cache_hits:  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        assert not dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
    else:
        assert dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        llvm = test_package_cmake.erf_libm_numba.inspect_llvm()
        assert "@erf" in llvm
        asm = test_package_cmake.erf_libm_numba.inspect_asm()
        assert f"{_PREFIX}erf" in asm
    # ruff: enable[private-member-access]

    with pytest.raises(kn.RedefinitionError):
        @kn.cdecl("c")
        def erf(x: np.float32) -> np.float64: ...

    with pytest.raises(kn.RedefinitionError):
        @kn.cdecl("m")
        def erf(arg: np.float64) -> np.float32: ...

    @kn.cdecl
    def erf(arg: np.float64) -> np.float64: ...

    assert erf is test_package_cmake.erf


@hypothesis.given(st.floats())
def test_cydecl(x: float) -> None:
    """Test `kn.cydecl`."""
    actual = test_package_cmake.erf_scipy_ctypes(x)
    desired = math.erf(x)
    np.testing.assert_allclose(actual, desired, rtol=1e-14)

    actual = test_package_cmake.erf_scipy_numba(x)
    assert isinstance(actual, float)
    desired = scipy.special.erf(x)
    np.testing.assert_equal(actual, desired)

    dispatcher = test_package_cmake.erf_scipy_numba.dispatcher
    # ruff: disable[private-member-access]
    cache = dispatcher._cache  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
    assert isinstance(cache, caching.FunctionCache)
    if dispatcher._cache_hits:  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        assert not dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
    else:
        assert dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        llvm = test_package_cmake.erf_scipy_numba.inspect_llvm()
        assert "@.const.__pyx_capi__" in llvm
        assert "@PyCapsule_GetPointer" in llvm
        asm = test_package_cmake.erf_scipy_numba.inspect_asm()
        assert f"{_PREFIX}.const.__pyx_capi__" in asm
        assert f"{_PREFIX}PyCapsule_GetPointer" in asm
    # ruff: enable[private-member-access]

    with pytest.raises(kn.RedefinitionError):
        @kn.cydecl("scipy.special.cython_special")
        def __pyx_fuse_1erf(
            arg: np.float64,
            skip_dispatch: np.uint32,
        ) -> np.float64: ...

    @kn.cydecl("scipy.special.cython_special")
    def __pyx_fuse_1erf(
            arg: np.float64,
            skip_dispatch: np.int32,
        ) -> np.float64: ...

    assert __pyx_fuse_1erf is test_package_cmake.erf_cydecl


def test_ccall() -> None:
    """Test `kn.ccall`."""
    name = np.frombuffer(b"Kimnara\0", np.uint8)
    a, b = test_package_cmake.hello(name)
    assert onp.is_array_1d(a, np.uint8)
    assert onp.is_array_1d(b, np.uint8)
    assert a[:-1].tobytes() == b"Hello, world!"
    assert b[:-1].tobytes() == b"Hello, Kimnara!"

    a, b = test_package_cmake.hello.__wrapped__(name)
    assert a[:-1].tobytes() == b"Hello, world!"
    assert b[:-1].tobytes() == b"Hello, Kimnara!"

    dispatcher = test_package_cmake.hello.dispatcher
    # ruff: disable[private-member-access]
    cache = dispatcher._cache  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
    assert isinstance(cache, caching.FunctionCache)
    if dispatcher._cache_hits:  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        assert not dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
    else:
        assert dispatcher._cache_misses  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        llvm = test_package_cmake.hello.inspect_llvm()
        assert "@sprintf" in llvm
        asm = test_package_cmake.hello.inspect_asm()
        assert f"{_PREFIX}sprintf" in asm
    # ruff: enable[private-member-access]


def test_cycall(
    recwarn: pytest.WarningsRecorder,
    rng: np.random.Generator,
) -> None:
    """Test `kn.cycall`."""
    a = rng.normal(size=(4, 2))
    a[:, 1] = 1
    b = rng.normal(size=4)
    desired = np.linalg.lstsq(a, b, rcond=-1)[0]
    # scipy.linalg.cython_lapack expects Fortran-ordered arrays
    actual = test_package_cmake.lstsq_ctypes(a.T, b)
    assert onp.is_array_1d(actual, np.float64)
    np.testing.assert_allclose(actual, desired, rtol=1e-14)

    # The arrays are copied internally to meet align="mkl"
    # We don't have to copy them ourselves
    actual = test_package_cmake.lstsq_numba(a.T, b)
    assert onp.is_array_1d(actual, np.float64)
    np.testing.assert_allclose(actual, desired, rtol=1e-14)

    assert not recwarn
    llvm = test_package_cmake.lstsq_numba.inspect_llvm()
    if not recwarn:
        assert "@.const.__pyx_capi__" in llvm
        assert "@PyCapsule_GetPointer" in llvm
        asm = test_package_cmake.lstsq_numba.inspect_asm()
        assert f"{_PREFIX}.const.__pyx_capi__" in asm
        assert f"{_PREFIX}PyCapsule_GetPointer" in asm
        assert not recwarn
