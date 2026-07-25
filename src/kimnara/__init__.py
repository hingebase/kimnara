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

"""Kimnara: Modern scientific computing framework."""

__all__ = [
    "AVX",
    "AVX512",
    "SSE",
    "A",
    "Alignment",
    "C",
    "Error",
    "F",
    "Mut",
    "Pad",
    "PerformanceWarning",
    "TypeInferenceError",
    "array",
    "asarray",
    "cfunc",
    "empty",
    "func",
    "gufunc",
    "isaligned",
    "quantity",
    "ufunc",
]

from . import align as align
from . import logging as logging
from . import threading as threading
from ._functions import cfunc, func, gufunc, ufunc
from ._quantity import quantity
from ._types import AVX, AVX512, SSE, A, Alignment, C, F, Mut, Pad
from .align import array, asarray, empty, isaligned
from .exceptions import (
    Error,
    PerformanceWarning,
    TypeInferenceError,
)
