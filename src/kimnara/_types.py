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

__all__ = ["AVX", "AVX512", "SSE", "A", "Alignment", "C", "F", "Pad"]

import contextlib
import enum
import functools
from typing import cast

import llvmlite.binding  # pyright: ignore[reportMissingTypeStubs]
from optype.typing import AnyComplex
from typing_extensions import final, override

import kimnara as kn

from . import _spec


@final
class Alignment(enum.Enum):
    A = _spec.Alignment("A")
    C = _spec.Alignment("C")
    F = _spec.Alignment("F")
    SSE = _spec.Alignment("SSE", multiple_of=16)
    AVX = _spec.Alignment("AVX", multiple_of=32)
    AVX512 = _spec.Alignment("AVX512", multiple_of=64)
    MKL = _spec.Alignment("MKL", multiple_of=64, not_multiple_of=4096)
    """Data alignment recommended by MKL.

    https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/current/coding-techniques.html
    """

    @property
    def allocator(self) -> contextlib.AbstractContextManager[object, None]:
        match self.value.multiple_of:
            case 32:
                return kn.align.AVXAllocator
            case 64:
                return kn.align.AVX512Allocator
            case _:
                return _sse_allocator

    @classmethod
    @override
    def _missing_(cls, value: object) -> "Alignment | None":
        if isinstance(value, str):
            if value == "host":
                return _host_alignment()
            value = value.upper()
            # https://www.phoronix.com/news/Intel-AVX10-Drops-256-Bit
            if value.startswith(("AVX512", "AVX-512", "AVX10")):
                return AVX512
            if value.startswith("AVX"):
                return AVX
            if value.startswith("SSE"):
                return SSE
            return cls.__members__.get(value)
        return None


# We don't expose kn.Alignment.MKL to outer namespace
# since the name `kn.MKL` would be confusing.
A = Alignment.A
C = Alignment.C
F = Alignment.F
SSE = Alignment.SSE
AVX = Alignment.AVX
AVX512 = Alignment.AVX512


@final
class Pad:
    __slots__ = ("value",)

    def __init__(self, value: AnyComplex | None) -> None:
        self.value = None if value is None else _spec.scalar(value)


@functools.lru_cache(maxsize=1)
def _host_alignment() -> Alignment:
    try:
        features = cast(
            "dict[str, bool]",
            llvmlite.binding.get_host_cpu_features(),
        )
    except RuntimeError:
        pass
    else:
        for k, v in features.items():
            if v and k.startswith("avx512"):
                return AVX512
        if features.get("avx"):
            return AVX
        if features.get("sse"):
            return SSE
    return C


_sse_allocator = contextlib.nullcontext()
