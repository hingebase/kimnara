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

import ctypes
import math
import sys

import numpy as np
import scipy

import kimnara as kn


def test_qsort() -> None:
    """Implement the `qsort` example from the `ctypes` docs in Numba.

    https://docs.python.org/3/library/ctypes.html#callback-functions
    """
    if sys.platform == "win32":
        libc = ctypes.cdll.ucrtbase
    else:
        libc = ctypes.CDLL(None)
    qsort = ctypes.CFUNCTYPE(
        None,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_ssize_t, ctypes.c_ssize_t),
    )(("qsort", libc))
    ia = np.array([5, 1, 7, 33, 99], np.intc)
    qsort(ia.ctypes, len(ia), ia.itemsize, _cmp_func)
    np.testing.assert_array_equal(ia, [1, 5, 7, 33, 99])


def test_quad() -> None:
    """Implement the `quad` example from the Numba docs in `ctypes`.

    https://numba.readthedocs.io/en/stable/user/cfunc.html#example
    """
    integrand = kn.cfunc(_integrand)._as_parameter_  # pyright: ignore[reportPrivateUsage]
    actual, abs_tol = scipy.integrate.quad(integrand, 1., math.inf)
    desired = .148495506776
    assert math.isclose(actual, desired)
    assert math.isclose(actual, desired, rel_tol=0, abs_tol=abs_tol)


@kn.cfunc(cache=False, nopython=True)
def _cmp_func(a: np.intp, b: np.intp) -> np.intc:
    pa = kn.ops.cast("kn.typing.CPointer[np.intc]", a)
    pb = kn.ops.cast("kn.typing.CPointer[np.intc]", b)
    return pa[0] - pb[0]


def _integrand(t: np.float64) -> np.float64:
    return scipy.special.expn(0, t) / t
