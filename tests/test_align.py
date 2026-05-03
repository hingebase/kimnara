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

"""Test kimnara.align API."""

import numpy as np
import pytest

import kimnara as kn


def test_allocator(subtests: pytest.Subtests) -> None:
    """Test direct use of allocator objects."""
    with subtests.test("32-byte alignment"), kn.align.AVXAllocator:
        assert _exactly_aligned(32)

    with subtests.test("64-byte alignment"), kn.align.AVX512Allocator:
        assert _exactly_aligned(64)

    with subtests.test("Default 16-byte alignment"):
        assert _exactly_aligned(16)


def _exactly_aligned(nbytes: int = 32) -> bool:
    arrs = [
        np.empty((2**i + j,), np.uint8) for i in range(22) for j in range(32)
    ]
    twice = 2 * nbytes
    return {arr.ctypes.data % twice for arr in arrs} == {0, nbytes}
