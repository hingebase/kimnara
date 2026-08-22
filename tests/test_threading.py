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

"""Test kimnara.threading API."""

import ctypes
import math
import os
import sys
import time
import warnings
from collections.abc import Callable
from typing import NoReturn, cast

import numba.core.registry  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import override

import kimnara as kn

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

_DUMMY_THREAD_ID = 42


def test_custom() -> None:
    """Test custom threading backend."""
    names = ["42", "prefer_openmp", "prefer_tbb"]
    if kn.threading.openmp_available():
        names.append("openmp")
    if kn.threading.tbb_available():
        names.append("tbb")

    for name in names:
        with pytest.raises(ValueError, match=f"backend name: {name!r}"):
            kn.threading.register_custom_backend(name, 0, 0, 0, np.vectorize)

    get_thread_id = ctypes.cast(_get_thread_id, ctypes.c_void_p).value
    assert get_thread_id

    numba_get_thread_id = cast(
        "numba.core.registry.CPUDispatcher",
        numba.njit(_numba_get_thread_id),  # pyright: ignore[reportUnknownMemberType]
    )
    kn.threading.register_custom_backend(
        "pytest", 0, get_thread_id, 0, _NumPyVectorize)
    with kn.threading.using_backend("pytest") as vectorize:
        assert vectorize is _NumPyVectorize
        assert numba_get_thread_id() == _DUMMY_THREAD_ID


def test_np_vectorize(subtests: pytest.Subtests) -> None:
    """Test direct use of `kn.threading.*NumPyVectorize` objects."""
    n = 6
    x = np.random.default_rng().random((n, 100))

    desired = np.vectorize(np.add.reduce, signature="(n)->()")(x)
    for vectorize in [
        kn.threading.OpenMPNumPyVectorize,
        kn.threading.TBBNumPyVectorize,
    ]:
        skip_parallel_test = None

        with subtests.test(
            "Results should be exactly equal",
            vectorize=vectorize,
        ):
            reduce = vectorize(np.add.reduce, signature="(n)->()")
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always", kn.PerformanceWarning)
                try:
                    actual = reduce(x)
                except kn.threading.BackendUnavailableError:
                    if os.getenv("PIXI_PROJECT_NAME") == "kimnara":
                        pytest.fail(
                            "Threading backends should always be available "
                            "when installed by Pixi",
                        )
                    if vectorize.backend_site_package_available():
                        pytest.fail(
                            "Threading backend was installed from PyPI "
                            "but not loaded",
                        )
                    continue
            for w in record:
                if isinstance(w, kn.PerformanceWarning):
                    skip_parallel_test = w.message
                else:
                    warnings.showwarning(
                        w.message,
                        w.category,
                        w.filename,
                        w.lineno,
                        w.file,
                        w.line,
                    )
            np.testing.assert_array_equal(actual, desired)

        if issubclass(vectorize, kn.threading.OpenMPNumPyVectorize):
            with subtests.test(
                "OpenMP backend should raise exception groups",
                vectorize=vectorize,
            ):
                throw = vectorize(_raise, signature="(n)->()")
                with pytest.raises(
                    ExceptionGroup,
                    check=lambda eg: len(eg.exceptions) == n and all(
                        isinstance(e, _ExpectedError) for e in eg.exceptions
                    ),
                ):
                    throw(x)
        else:
            with subtests.test(
                "TBB backend should propagate exceptions as-is",
                vectorize=vectorize,
            ):
                throw = vectorize(_raise, signature="(n)->()")
                with pytest.raises(_ExpectedError):
                    throw(x)

        with subtests.test(
            "The wrapped function should run concurrently",
            vectorize=vectorize,
        ):
            match skip_parallel_test:
                case None:
                    pass
                case str():
                    pytest.skip(skip_parallel_test)
                case _:
                    pytest.skip(skip_parallel_test.args[0])
            sleep = vectorize(_sleep, signature="(n)->()")
            tic = time.monotonic()
            sleep(x)
            toc = time.monotonic()
            eps = toc - math.nextafter(toc, -math.inf)
            assert 1-eps < toc-tic < n-eps  # ruff: ignore[missing-whitespace-around-arithmetic-operator]


class _ExpectedError(Exception):
    pass


class _NumPyVectorize(kn.threading.TBBNumPyVectorize):
    @override
    def np_vectorize_impl(self, func: Callable[[int], None], n: int) -> None:
        raise _ExpectedError


@ctypes.CFUNCTYPE(ctypes.c_ssize_t)
def _get_thread_id() -> int:
    return _DUMMY_THREAD_ID


def _numba_get_thread_id() -> int:
    return numba.get_thread_id()  # pyright: ignore[reportPrivateImportUsage]


def _raise(_: npt.NDArray[np.float64]) -> NoReturn:
    raise _ExpectedError


def _sleep(_: npt.NDArray[np.float64]) -> None:
    time.sleep(1)
