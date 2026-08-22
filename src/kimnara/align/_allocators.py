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

__all__ = ["AVX512Allocator", "AVXAllocator"]

import ctypes
from typing import TYPE_CHECKING, Any

from typing_extensions import CapsuleType

from . import _mimalloc

if not TYPE_CHECKING:
    from numpy_allocator import (
        type as _AllocatorMeta,  # ruff: ignore[lowercase-imported-as-non-lowercase]
    )
else:
    from _typeshed import Unused
    from numpy.typing import NDArray

    class _AllocatorMeta(type):
        # Returns the allocator type object itself, but cannot be
        # annotated with `typing_extensions.Self` in metaclass
        def __enter__(cls) -> object: ...

        def __exit__(cls, *excinfo: Unused) -> None: ...
        def handler(cls) -> CapsuleType: ...
        def handles(cls, array: NDArray[Any], /) -> bool: ...

_POINTER_SIZE = ctypes.sizeof(ctypes.c_void_p)


class AVXAllocator(metaclass=_AllocatorMeta):
    _calloc_ = _mimalloc.calloc_funcs
    _free_ = _mimalloc.free_funcs
    _malloc_ = _mimalloc.malloc_funcs
    _realloc_ = _mimalloc.realloc_funcs


class AVX512Allocator(metaclass=_AllocatorMeta):
    _calloc_ = _mimalloc.calloc_funcs + _POINTER_SIZE
    _free_ = _mimalloc.free_funcs
    _malloc_ = _mimalloc.malloc_funcs + _POINTER_SIZE
    _realloc_ = _mimalloc.realloc_funcs + _POINTER_SIZE


_mimalloc.set_destructor(AVXAllocator.handler())
_mimalloc.set_destructor(AVX512Allocator.handler())
