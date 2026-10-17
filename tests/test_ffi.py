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
import scipy.special
import test_package_cmake
from numba.core import caching  # pyright: ignore[reportMissingTypeStubs]


@hypothesis.given(st.floats())
def test_cdecl(x: float) -> None:
    """Test `kn.cdecl`."""
    actual = test_package_cmake.erf(x)
    desired = math.erf(x)
    np.testing.assert_equal(actual, desired, strict=True)

    actual = test_package_cmake.erf_libm_numba(x)
    np.testing.assert_equal(actual, desired, strict=True)

    dispatcher = test_package_cmake.erf_libm_numba.dispatcher
    # ruff: disable[private-member-access]
    cache = dispatcher._cache  # pyright: ignore[reportPrivateUsage]
    assert isinstance(cache, caching.FunctionCache)
    if dispatcher._cache_hits:  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        assert not dispatcher._cache_misses  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    else:
        assert dispatcher._cache_misses  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        llvm = test_package_cmake.erf_libm_numba.inspect_llvm()
        assert "@erf" in llvm
        asm = test_package_cmake.erf_libm_numba.inspect_asm()
        prefix = "$_" if sys.platform == "darwin" else "$"
        assert f"{prefix}erf" in asm
    # ruff: enable[private-member-access]


@hypothesis.given(st.floats())
def test_cydecl(x: float) -> None:
    """Test `kn.cydecl`."""
    actual = test_package_cmake.erf_scipy_ctypes(x)
    desired = math.erf(x)
    np.testing.assert_allclose(actual, desired, rtol=1e-14, strict=True)

    actual = test_package_cmake.erf_scipy_numba(x)
    desired = scipy.special.erf(x)
    np.testing.assert_equal(actual, desired, strict=True)

    dispatcher = test_package_cmake.erf_scipy_numba.dispatcher
    # ruff: disable[private-member-access]
    cache = dispatcher._cache  # pyright: ignore[reportPrivateUsage]
    assert isinstance(cache, caching.FunctionCache)
    if dispatcher._cache_hits:  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        assert not dispatcher._cache_misses  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
    else:
        assert dispatcher._cache_misses  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        llvm = test_package_cmake.erf_scipy_numba.inspect_llvm()
        assert "@.const.__pyx_capi__" in llvm
        assert "@PyCapsule_GetPointer" in llvm
        asm = test_package_cmake.erf_scipy_numba.inspect_asm()
        prefix = "$_" if sys.platform == "darwin" else "$"
        assert f"{prefix}.const.__pyx_capi__" in asm
        assert f"{prefix}PyCapsule_GetPointer" in asm
    # ruff: enable[private-member-access]
