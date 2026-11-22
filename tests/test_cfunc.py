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

"""Test `kn.cfunc`."""

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
import test_package_cmake
from numba.core import caching  # pyright: ignore[reportMissingTypeStubs]

import kimnara as kn

if TYPE_CHECKING:
    from kimnara._functions._func import NumbaFunc


def test_call_cfunc_in_numba() -> None:
    """`kn.cfunc` objects should be callable from Numba."""
    expected = 42
    assert test_package_cmake.call_cfunc_in_numba() == expected
    _ensure_cache_enabled(test_package_cmake.call_cfunc_in_numba)


def test_qsort() -> None:
    """Implement the `qsort` example from the `ctypes` docs in Numba.

    https://docs.python.org/3/library/ctypes.html#callback-functions
    """
    ia = np.array([5, 1, 7, 33, 99], np.intc)
    test_package_cmake.qsort(ia)
    expected = np.array([1, 5, 7, 33, 99], np.intc)
    np.testing.assert_array_equal(ia, expected, strict=True)
    _ensure_cache_enabled(test_package_cmake.qsort)


def test_quad() -> None:
    """Implement the `quad` example from the Numba docs in `ctypes`.

    https://numba.readthedocs.io/en/stable/user/cfunc.html#example
    """
    integrand = kn.cfunc(_integrand)._as_parameter_  # pyright: ignore[reportPrivateUsage]
    actual, abs_tol = scipy.integrate.quad(integrand, 1., math.inf)
    desired = .148495506776
    assert math.isclose(actual, desired)
    assert math.isclose(actual, desired, rel_tol=0, abs_tol=abs_tol)


def _ensure_cache_enabled(func: "NumbaFunc[Any]") -> None:
    assert isinstance(func.dispatcher._cache, caching.FunctionCache)  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]  # ruff: ignore[private-member-access]


def _integrand(t: np.float64) -> np.float64:
    return scipy.special.expn(0, t) / t
