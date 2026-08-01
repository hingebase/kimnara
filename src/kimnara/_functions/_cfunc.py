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

__all__ = ["NumbaCFunc", "NumbaT", "PyCFunc", "PythonT"]

import abc
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numba.core import types  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    templates,
)
from typing_extensions import TypeVar, override

import kimnara as kn
from kimnara import _spec

from . import _common

if TYPE_CHECKING:
    from ctypes import _CFunctionType  # pyright: ignore[reportPrivateUsage]

NumbaT = TypeVar(
    "NumbaT",
    np.bool_,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
    bool, int, float, complex, None,
)
_T = TypeVar("_T")

if sys.version_info >= (3, 14):
    PythonT = TypeVar(
        "PythonT",
        np.bool_,
        np.int8, np.int16, np.int32, np.int64, np.intp,
        np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
        np.float32, np.float64, np.complex64, np.complex128,
        bool, int, float, complex, None,
    )
else:
    PythonT = TypeVar(
        "PythonT",
        np.bool_,
        np.int8, np.int16, np.int32, np.int64, np.intp,
        np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
        np.float32, np.float64,
        bool, int, float, None,
    )


class _CFunc(types.WrapperAddressProtocol, _common.Inferable[_T]):
    def __init__(self, wrapped: Callable[..., _T]) -> None:
        self.infer(wrapped)

    @override
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_scalar=True,
        )

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_none=True,
            allow_scalar=True,
        )

    @property
    @abc.abstractmethod
    def _as_parameter_(self) -> "_CFunctionType":
        raise NotImplementedError

    def __call__(self, *args: object) -> _T:
        return self.__wrapped__(*args)


class NumbaCFunc(_CFunc[NumbaT], _common.Dispatchable):
    @property
    @override
    def _as_parameter_(self) -> "_CFunctionType":
        raise NotImplementedError

    @override
    def signature(self) -> templates.Signature:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError

    @override
    def __wrapper_address__(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError


class PyCFunc(_CFunc[PythonT]):
    @property
    @override
    def _as_parameter_(self) -> "_CFunctionType":
        raise NotImplementedError

    @override
    def signature(self) -> templates.Signature:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError

    @override
    def __wrapper_address__(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError
