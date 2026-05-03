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
    "TBBNumPyVectorize",
    "tbb_available",
]

from collections.abc import Callable

from typing_extensions import override

from . import _genericpool

with _genericpool.add_dll_directory():
    try:
        from . import _tbb
    except ImportError:
        _tbb = None


class TBBNumPyVectorize(_genericpool.NumPyVectorize):
    @classmethod
    @override
    def backend_site_package_name(cls) -> str:
        return "tbb"

    @override
    def np_vectorize_impl(self, func: Callable[[int], None], n: int) -> None:
        if not _tbb:
            self.backend_unavailable("oneTBB runtime is not installed")
        _genericpool.check_num_threads(_tbb)
        _tbb.np_vectorize(func, n)


def tbb_available() -> bool:
    return bool(_tbb)
