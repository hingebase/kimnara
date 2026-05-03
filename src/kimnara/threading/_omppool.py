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
    "OpenMPNumPyVectorize",
    "openmp_available",
]

import sys
from collections.abc import Callable

from typing_extensions import override

from . import _genericpool

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

with _genericpool.add_dll_directory():
    try:
        from . import _openmp
    except ImportError:
        _openmp = None


class OpenMPNumPyVectorize(_genericpool.NumPyVectorize):
    @classmethod
    @override
    def backend_site_package_name(cls) -> str:
        return "intel-openmp"

    @override
    def np_vectorize_impl(self, func: Callable[[int], None], n: int) -> None:
        if not _openmp:
            self.backend_unavailable("LLVM OpenMP runtime is not installed")
        _genericpool.check_num_threads(_openmp)
        excs: list[BaseException] = []
        _openmp.np_vectorize(func, n, excs)
        if excs:
            message = "Error calling OpenMP parallelized function"
            raise BaseExceptionGroup(message, excs)


def openmp_available() -> bool:
    return bool(_openmp)
