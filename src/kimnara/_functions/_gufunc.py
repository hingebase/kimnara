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
    "NumbaGUFunc",
    "NumbaParallelGUFunc",
    "NumbaType",
    "PyGUFunc",
    "PythonType",
]

from typing import TypeAlias

from typing_extensions import TypeVar, Unpack, override

import kimnara as kn
from kimnara import _spec
from kimnara._typing import ArrayLike, Input, Outputs, Scalar, UFuncKwargs

from . import _common

NumbaType: TypeAlias = None
PythonType = ArrayLike[Scalar] | tuple[ArrayLike[Scalar], ...]
_T = TypeVar("_T", NumbaType, PythonType)


class _GUFunc(_common.UFuncCompiler[_T]):
    @property
    @override
    def identity(self) -> None:
        pass

    @override
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=align,
            allow_align=frozenset(a for a in kn.Alignment if a is not kn.F),
            allow_array=True,
            allow_scalar=True,
            allow_units=True,
            pad_value=pad_value,
            readonly=True,
        )


class _NumbaGUFuncBase(_GUFunc[NumbaType], _common.UFuncWrapper[NumbaType]):
    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        raise NotImplementedError

    @override
    def infer_impl(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> tuple[tuple[_spec.Type, ...], _spec.Type]:
        raise NotImplementedError

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=align,
            allow_align=frozenset(a for a in kn.Alignment if a is not kn.F),
            allow_array=True,
            allow_units=True,
            pad_value=pad_value,
        )


class NumbaGUFunc(_NumbaGUFuncBase, _common.Dispatchable):
    pass


class NumbaParallelGUFunc(_NumbaGUFuncBase):
    pass


class PyGUFunc(_GUFunc[PythonType]):
    @property
    @override
    def nin(self) -> int:
        raise NotImplementedError

    @property
    @override
    def nout(self) -> int:
        raise NotImplementedError

    @property
    @override
    def types(self) -> list[str]:
        raise NotImplementedError

    @property
    @override
    def signature(self) -> str:
        raise NotImplementedError

    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        raise NotImplementedError

    @override
    def infer_impl(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> tuple[tuple[_spec.Type, ...], _spec.Type]:
        raise NotImplementedError

    @override
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        return _spec.TypingContext(
            align=kn.A,
            allow_align=frozenset([kn.A]),
            allow_array=True,
            allow_scalar=True,
            allow_tuple=True,
            allow_units=True,
        )
