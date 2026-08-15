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
    "BooleanOutputT",
    "NumbaGUFunc",
    "NumbaParallelGUFunc",
    "OutputT",
    "OutputsT",
    "PyGUFunc",
]

import abc
import contextlib
import functools
import inspect
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast

import numba.core.types  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import numpy.typing as npt
import optype as op
import optype.numpy as onp
from numba.np.ufunc import (  # pyright: ignore[reportMissingTypeStubs]
    gufunc,
    sigparse,
    ufuncbuilder,
)
from numpy_typing_compat import NUMPY_GE_2_0
from pint.facets.numpy.quantity import NumpyQuantity
from typing_extensions import Any, TypeVar, Unpack, overload, override

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._typing import (
    ArrayLike,
    FastMathOptions,
    Input,
    Number,
    Output,
    Outputs,
    Scalar,
    UFuncKwargs,
)

from . import _common

if TYPE_CHECKING:
    from kimnara._spec._numpy._base import ScalarType, Type

_T = TypeVar("_T")

BooleanOutputT = TypeVar(
    "BooleanOutputT",
    bound=onp.ToJustBool | npt.NDArray[np.bool_],
)
OutputT = TypeVar(
    "OutputT",
    bound=op.JustInt | op.JustFloat | op.JustComplex | ArrayLike[Number],
)
OutputsT = TypeVar("OutputsT", bound=tuple[complex | ArrayLike[Scalar], ...])

_COPY = cast("bool", None) if NUMPY_GE_2_0 else False


class _GUFunc(_common.UFuncCompiler[_T]):
    def __init__(
        self,
        wrapped: Callable[..., _T],
        signature: str,
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: op.typing.AnyComplex | None = None,
    ) -> None:
        self.inputs, self.outputs = cast(
            "tuple[list[tuple[str, ...]], list[tuple[str, ...]]]",
            sigparse.parse_signature(signature),  # pyright: ignore[reportUnknownMemberType]
        )
        self.infer(wrapped, align=align, pad_value=pad_value)

    @property
    @override
    def identity(self) -> None:
        pass

    @overload
    def __call__(
        self: "_GUFunc[None]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> Outputs: ...

    @overload
    def __call__(
        self: "_GUFunc[BooleanOutputT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[np.bool_]: ...

    @overload
    def __call__(
        self: "_GUFunc[OutputT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> ArrayLike[Number] | NumpyQuantity[ArrayLike[Number]]: ...

    @overload
    def __call__(
        self: "_GUFunc[OutputsT]",
        *args: Input,
        **kwargs: Unpack[UFuncKwargs],
    ) -> tuple[Output, ...]: ...

    @override
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
        if "where" in kwargs:
            message = (
                f"{self.__name__}() got an unexpected keyword argument 'where'"
            )
            raise TypeError(message)
        match kwargs.get("order", "K"):
            case "K":
                order = "K"
            case "F":
                message = "order='F' is unsupported in `kn.gufunc`"
                raise ValueError(message)
            case _:
                # "A" means ("F" or "C"), and "F" is already blocked
                order = "C"
        wrapper = self.wrap_call(order, subok=kwargs.get("subok", True))
        return wrapper(*args)

    @override
    def infer_impl(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> tuple[tuple[_spec.Type, ...], _spec.Type]:
        if align is kn.F:
            message = "align='F' is unsupported in `kn.gufunc`"
            raise kn.TypeInferenceError(message)
        it = iter(self.sig.parameters.values())

        ctx = self.input_context(align, pad_value)
        args: list[_spec.Type] = []
        for ndim, v in zip(
            map(len, self.inputs),
            it,
            strict=isinstance(self, PyGUFunc),
        ):
            if v.POSITIONAL_ONLY is not v.kind is not v.POSITIONAL_OR_KEYWORD:
                message = "kwargs and varargs are disallowed"
                raise TypeError(message)
            if v.default is not v.empty:
                message = "default parameter values are disallowed"
                raise TypeError(message)
            args.append(ctx._replace(ndim=ndim).infer(v.annotation))

        ctx = self.output_context(align, pad_value)
        restype = self.infer_output(ctx, it)
        with contextlib.suppress(ValueError):
            [restype] = restype
        return tuple(args), restype

    @abc.abstractmethod
    def infer_output(
        self,
        ctx: _spec.TypingContext,
        params: Iterator[inspect.Parameter],
    ) -> _spec.Type:
        raise NotImplementedError

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

    @abc.abstractmethod
    def wrap_call(
        self,
        order: Literal["C", "K"],
        *,
        subok: bool = True,
    ) -> Callable[..., Any]:
        raise NotImplementedError


class _NumbaGUFuncBase(_GUFunc[None], _common.UFuncWrapper[None], Generic[_T]):
    @override
    def __init__(
        self,
        wrapped: Callable[..., None],
        signature: str,
        *,
        align: "str | kn.Alignment" = "A",
        boundscheck: bool | None = None,
        cache: bool | None = None,
        fastmath: FastMathOptions = False,
        pad_value: op.typing.AnyComplex | None = None,
        parallel: _T = False,
    ) -> None:
        super().__init__(
            wrapped,
            signature,
            align=align,
            pad_value=pad_value,
        )
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = [arg.to_numba() for arg in self.restype]
        decorator = numba.guvectorize(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            [numba.core.types.void(*argtypes, *restype)],
            signature,
            boundscheck=boundscheck,
            cache=_common.can_cache(wrapped, cache=cache),
            fastmath=fastmath,
            target="parallel" if parallel else "cpu",
        )
        self.build(decorator, parallel)  # pyright: ignore[reportUnknownArgumentType]

        it = iter(self.sig.parameters.values())
        parameters = list(itertools.islice(it, len(argtypes)))
        if len(restype) == 1:
            return_annotation = next(it).annotation
        else:
            return_annotation = tuple[tuple(arg.annotation for arg in it)]
        self._sig = inspect.Signature(
            parameters,
            return_annotation=return_annotation,
        )

    @override
    def infer_output(
        self,
        ctx: _spec.TypingContext,
        params: Iterator[inspect.Parameter],
    ) -> _spec.Type:
        args: list[_spec.Type] = []
        for ndim, v in zip(map(len, self.outputs), params, strict=True):
            if v.POSITIONAL_ONLY is not v.kind is not v.POSITIONAL_OR_KEYWORD:
                message = "kwargs and varargs are disallowed"
                raise TypeError(message)
            if v.default is not v.empty:
                message = "default parameter values are disallowed"
                raise TypeError(message)
            args.append(ctx._replace(ndim=max(ndim, 1)).infer(v.annotation))

        # Numba gufunc must return None
        ctx = _spec.TypingContext(align=kn.A, allow_none=True)
        ctx.infer(self.sig.return_annotation)

        return ctx.make_tuple(args)

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

    @override
    def wrap_call(
        self,
        order: Literal["C", "K"],
        *,
        subok: bool = True,
    ) -> Callable[..., Any]:
        match self._fast_path:
            case True:
                ufunc = self._ufunc

                def wrapper(
                    *args: ArrayLike[Scalar],
                ) -> npt.NDArray[Scalar] | tuple[npt.NDArray[Scalar], ...]:
                    return ufunc(*args, order="C", subok=subok)
            case False:
                def wrapper(
                    *args: ArrayLike[Scalar],
                ) -> npt.NDArray[Scalar] | tuple[npt.NDArray[Scalar], ...]:
                    shapes = self._calculate_shapes(args)
                    out: list[npt.NDArray[Scalar]] = []
                    for arg, shape in zip(self.restype, shapes, strict=True):
                        x = cast("Type", arg).empty(shape)
                        if order == "C" and not x.flags.c_contiguous:
                            message = "order='C' conflicts with SIMD alignment"
                            raise ValueError(message)
                        out.append(x)
                    return self._ufunc(*args, *out)
            case None:
                def wrapper(
                    *args: ArrayLike[Scalar],
                ) -> npt.NDArray[Scalar] | tuple[npt.NDArray[Scalar], ...]:
                    shapes = self._calculate_shapes(args)
                    if self._retry_fast_path(shapes, order):
                        return self._ufunc(*args, order="C", subok=subok)
                    out = [
                        cast("Type", arg).empty(shape) for arg, shape in zip(
                            self.restype,
                            shapes,
                            strict=True,
                        )
                    ]
                    return self._ufunc(*args, *out)
        return self.validate_call(wrapper, self._sig)

    @abc.abstractmethod
    def build(
        self,
        decorator: Callable[[Callable[..., None]], Any],
        name: _T,
    ) -> None:
        raise NotImplementedError

    def _calculate_shapes(
        self,
        args: tuple[ArrayLike[Scalar], ...],
    ) -> Sequence[Sequence[int]]:
        # ruff: disable[private-member-access]
        broadcast_shape, dim_sizes = (
            _utils.function_base._parse_input_dimensions(args, self.inputs)  # pyright: ignore[reportPrivateUsage]
        )
        return _utils.function_base._calculate_shapes(  # pyright: ignore[reportPrivateUsage]
            broadcast_shape, dim_sizes, self.outputs)
        # ruff: enable[private-member-access]

    @functools.cached_property
    def _fast_path(self) -> bool | None:
        res = True
        for arg in self.restype:
            match getattr(arg, "_align", kn.A):
                case kn.A | kn.C:
                    pass
                case kn.SSE:
                    if cast("Type", arg).dtype is not np.complex128:
                        res = None
                case _:
                    return False
        return res

    def _retry_fast_path(
        self,
        shapes: Sequence[Sequence[int]],
        order: Literal["C", "K"],
    ) -> bool:
        align = kn.SSE
        for arg, shape in zip(self.restype, shapes, strict=True):
            if (
                getattr(arg, "_align", None) is align
                and _utils.calculate_padding(
                    shape,
                    cast("Type", arg).dtype,
                    align.value,
                )
            ):
                if order == "C":
                    message = "order='C' conflicts with SSE alignment"
                    raise ValueError(message)
                return False
        return True


class NumbaGUFunc(_NumbaGUFuncBase[Literal[False]], _common.Dispatchable):
    @override
    def build(
        self,
        decorator: Callable[[Callable[..., None]], gufunc.GUFunc],
        name: Literal[False],
    ) -> None:
        ufunc = decorator(self.__wrapped__)
        self.dispatcher = cast(
            "ufuncbuilder.UFuncDispatcher",
            ufunc.gufunc_builder.nb_func,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        )
        self._ufunc = cast("np.ufunc", ufunc.ufunc)


class NumbaParallelGUFunc(_NumbaGUFuncBase[str | Literal[True]]):
    @override
    def build(
        self,
        decorator: Callable[[Callable[..., None]], np.ufunc],
        name: str | Literal[True],
    ) -> None:
        with kn.threading.using_backend(name):
            self._ufunc = decorator(self.__wrapped__)


class PyGUFunc(_GUFunc[_T]):
    @override
    def __init__(
        self,
        wrapped: Callable[..., _T],
        signature: str,
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: op.typing.AnyComplex | None = None,
        parallel: bool | str = False,
    ) -> None:
        super().__init__(
            wrapped,
            signature,
            align=align,
            pad_value=pad_value,
        )
        otypes = [cast("ScalarType", arg).dtype for arg in self.restype]
        if parallel:
            with kn.threading.using_backend(parallel) as vectorize:
                ufunc = vectorize(wrapped, otypes, signature=signature)
        else:
            ufunc = np.vectorize(wrapped, otypes, signature=signature)
        self._ufunc = ufunc

    @property
    @override
    def signature(self) -> str:
        return self._ufunc.signature or _utils.unreachable()

    @override
    def infer_output(
        self,
        ctx: _spec.TypingContext,
        params: Iterator[inspect.Parameter],
    ) -> _spec.Type:
        try:
            restype = ctx.infer(self.sig.return_annotation)
        except StopIteration as e:
            message = "nout more than signature"
            raise kn.TypeInferenceError(message) from e
        for _ in cast("Iterator[int]", ctx.ndim):
            message = "nout less than signature"
            raise kn.TypeInferenceError(message)
        return restype

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
            ndim=map(len, self.outputs),
        )

    @override
    def wrap_call(
        self,
        order: Literal["C", "K"],
        *,
        subok: bool = True,
    ) -> Callable[..., Any]:
        ufunc = self._ufunc
        if len(self.restype) == 1:
            @self.validate_call
            def wrapper(*args: ArrayLike[Scalar]) -> npt.NDArray[Scalar]:
                out = ufunc(*args)
                return np.array(out, copy=_COPY, order=order, subok=subok)
        else:
            @self.validate_call
            def wrapper(
                *args: ArrayLike[Scalar],
            ) -> tuple[npt.NDArray[Scalar], ...]:
                out = ufunc(*args)
                return tuple(
                    np.array(x, copy=_COPY, order=order, subok=subok)
                    for x in out
                )
        return wrapper
