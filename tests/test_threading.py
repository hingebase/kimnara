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

import math
import os
import sys
import time
import warnings
from typing import NoReturn

import numpy as np
import numpy.typing as npt
import pytest

import kimnara as kn

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


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
            assert 1-eps < toc-tic < n-eps  # noqa: E226


class _ExpectedError(Exception):
    pass


def _raise(_: npt.NDArray[np.float64]) -> NoReturn:
    raise _ExpectedError


def _sleep(_: npt.NDArray[np.float64]) -> None:
    time.sleep(1)
