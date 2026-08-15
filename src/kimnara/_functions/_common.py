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
    "BaseUFuncWrapper",
    "Dispatchable",
    "Inferable",
    "UFuncCompiler",
    "UFuncWrapper",
    "Validator",
    "can_cache",
]

import abc
import collections
import functools
import inspect
import itertools
import linecache
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Literal, cast, no_type_check

import numpy as np
import pint.registry_helpers
import pydantic
import pydantic_core
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    ccallback,
    compiler,
    registry,
)
from numba.core.codegen import (  # pyright: ignore[reportMissingTypeStubs]
    _CFG,  # pyright: ignore[reportPrivateUsage]
)
from numba.core.dispatcher import (  # pyright: ignore[reportMissingTypeStubs]
    Dispatcher,
)
from numba.core.funcdesc import (  # pyright: ignore[reportMissingTypeStubs]
    FunctionDescriptor,
)
from numba.np.ufunc import (  # pyright: ignore[reportMissingTypeStubs]
    ufuncbuilder,
)
from optype.typing import AnyComplex
from typing_extensions import (
    Any,
    LiteralString,
    Protocol,
    TypeVar,
    Unpack,
    override,
)

import kimnara as kn
from kimnara import _spec
from kimnara._typing import Input, Outputs, UFuncKwargs

if TYPE_CHECKING:
    from _typeshed import Unused
    from numpy import _CastingKind  # pyright: ignore[reportPrivateUsage]

    from kimnara._spec._numpy._base import ScalarType

_T = TypeVar("_T")


class _DisasmCFG(Protocol):
    def _repr_svg_(self) -> str: ...


class Dispatchable:
    @property
    def dispatcher(self) -> Dispatcher:
        return self._dispatcher

    @dispatcher.setter
    def dispatcher(
        self,
        value: Dispatcher | ccallback.CFunc | ufuncbuilder.UFuncDispatcher,
    ) -> None:
        if isinstance(value, Dispatcher):
            self._dispatcher = value
            return
        self._dispatcher = new = Dispatcher.__new__(Dispatcher)
        if isinstance(value, ufuncbuilder.UFuncDispatcher):
            new.overloads = value.overloads  # pyright: ignore[reportAttributeAccessIssue]
            new.py_func = value.py_func  # pyright: ignore[reportUnknownMemberType]
            return
        # ruff: disable[private-member-access]
        key = value._sig  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
        cres = value._cache.load_overload(  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
            key,
            registry.cpu_target.target_context,
        )
        if not cres:
            fndesc = FunctionDescriptor.__new__(FunctionDescriptor)
            fndesc.mangled_name = str(value.native_name).removeprefix("cfunc.")  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            cres = compiler.compile_result(  # pyright: ignore[reportUnknownMemberType]
                fndesc=fndesc,
                library=value._library,  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownArgumentType, reportUnknownMemberType]
            )
        new.overloads = collections.OrderedDict([(key, cres)])  # pyright: ignore[reportUnknownArgumentType]
        new.py_func = value._pyfunc  # pyright: ignore[reportAttributeAccessIssue, reportPrivateUsage, reportUnknownMemberType]
        # ruff: enable[private-member-access]

    @no_type_check
    def inspect_asm(self) -> str:
        [asm] = self._dispatcher.inspect_asm().values()
        return asm

    @no_type_check
    def inspect_cfg(self, **kwargs: object) -> _CFG:
        [cfg] = self._dispatcher.inspect_cfg(**kwargs).values()
        return cfg

    @no_type_check
    def inspect_disasm_cfg(self) -> _DisasmCFG:
        [disasm_cfg] = self._dispatcher.inspect_disasm_cfg().values()
        return disasm_cfg

    @no_type_check
    def inspect_llvm(self) -> str:
        [llvm] = self._dispatcher.inspect_llvm().values()
        return llvm


class Inferable(abc.ABC, Generic[_T]):
    def infer(
        self,
        wrapped: Callable[..., _T],
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: AnyComplex | None = None,
    ) -> None:
        self.sig = inspect.signature(wrapped, eval_str=True)
        self.__wrapped__ = wrapped
        self.__module__ = wrapped.__module__
        align = kn.Alignment(align)
        if pad_value is not None:
            pad_value = _spec.scalar(pad_value)
        self.argtypes, self.restype = self.infer_impl(align, pad_value)

    def infer_impl(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> tuple[tuple[_spec.Type, ...], _spec.Type]:
        ctx = self.input_context(align, pad_value)
        args: list[_spec.Type] = []
        for v in self.sig.parameters.values():
            if v.POSITIONAL_ONLY is not v.kind is not v.POSITIONAL_OR_KEYWORD:
                message = "kwargs and varargs are disallowed"
                raise TypeError(message)
            if v.default is not v.empty:
                message = "default parameter values are unsupported yet"
                raise NotImplementedError(message)
            arg = ctx.infer_with_diagnostics(
                v.annotation,
                "Failed to infer the type of argument {0.name!r}: {1}",
                v,
            )
            args.append(arg)
        ctx = self.output_context(align, pad_value)
        res = ctx.infer_with_diagnostics(self.sig.return_annotation)
        return tuple(args), res

    @abc.abstractmethod
    def input_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        raise NotImplementedError

    @abc.abstractmethod
    def output_context(
        self,
        align: "kn.Alignment",
        pad_value: complex | None,
    ) -> _spec.TypingContext:
        raise NotImplementedError

    def __reduce__(self) -> str:
        return self.__wrapped__.__name__


class Validator(Inferable[_T]):
    def alternative_signature(self) -> inspect.Signature:
        raise NotImplementedError

    def validate_call(
        self,
        wrapped: Callable[..., object],
        *,
        alternative: bool = False,
    ) -> Callable[..., Any]:
        if alternative:
            sig, pint_decorator = self._alternative_signature
        else:
            sig, pint_decorator = self._call_signature
        if wrapped is not self.__wrapped__:
            wrapped.__signature__ = sig  # pyright: ignore[reportFunctionMemberAccess]
        wrapper = pint_decorator(wrapped)
        wrapper.__signature__ = sig  # pyright: ignore[reportFunctionMemberAccess]
        wrapper.__annotations__ = annotations = {
            k: v.annotation for k, v in sig.parameters.items()
        }
        annotations["return"] = sig.return_annotation
        return _with_parameter_names(pydantic.validate_call(wrapper), sig)

    @functools.cached_property
    def _alternative_signature(self) -> tuple[
        inspect.Signature,
        Callable[[Callable[..., object]], Callable[..., object]],
    ]:
        sig = self.alternative_signature()
        return sig, self._pint_decorator(sig)

    @functools.cached_property
    def _call_signature(self) -> tuple[
        inspect.Signature,
        Callable[[Callable[..., object]], Callable[..., object]],
    ]:
        sig = inspect.Signature(
            parameters=[
                param.replace(
                    kind=inspect.Parameter.POSITIONAL_ONLY,
                    annotation=arg.to_python(),
                )
                for arg, param in zip(
                    self.argtypes,
                    self.sig.parameters.values(),
                    strict=True,
                )
            ],
            return_annotation=self.restype.to_python(),
        )
        return sig, self._pint_decorator(sig)

    def _pint_decorator(
        self,
        sig: inspect.Signature,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        return pint.registry_helpers.wraps(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            _spec.ureg,  # pyright: ignore[reportArgumentType]
            ret=self.restype.to_units(),
            args=(None,) * len(sig.parameters),  # Handled by pydantic
        )


class UFuncCompiler(Validator[_T]):
    @override
    def infer(
        self,
        wrapped: Callable[..., _T],
        *,
        align: "str | kn.Alignment" = "A",
        pad_value: AnyComplex | None = None,
    ) -> None:
        super().infer(wrapped, align=align, pad_value=pad_value)
        self.__qualname__ = wrapped.__qualname__  # This must not be a property

    @property
    def __name__(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.__wrapped__.__name__

    @property
    def __doc__(self) -> str | None:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.__wrapped__.__doc__

    @property
    def nin(self) -> int:
        return len(self.argtypes)

    @property
    def nout(self) -> int:
        return len(self.restype)

    @property
    def nargs(self) -> int:
        return self.nin + self.nout

    @property
    def ntypes(self) -> Literal[1]:
        return 1

    @property
    def types(self) -> list[str]:
        argtypes = "".join(
            np.dtype(cast("ScalarType", arg).dtype).char
            for arg in self.argtypes
        )
        restype = "".join(
            np.dtype(cast("ScalarType", arg).dtype).char
            for arg in self.restype
        )
        return [f"{argtypes}->{restype}"]

    @property
    @abc.abstractmethod
    def identity(self) -> Literal[0, 1] | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def signature(self) -> str | None:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(
        self,
        *args: Input[Any, Any],
        **kwargs: Unpack[UFuncKwargs],
    ) -> Outputs:
        raise NotImplementedError


class BaseUFuncWrapper(UFuncCompiler[_T]):
    _ufunc: np.ufunc

    @property
    @override
    def __doc__(self) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.__wrapped__.__doc__ or self._ufunc.__doc__

    @property
    @override
    def nin(self) -> int:
        return self._ufunc.nin

    @property
    @override
    def nout(self) -> int:
        return self._ufunc.nout

    @property
    @override
    def nargs(self) -> int:
        return self._ufunc.nargs

    @property
    @override
    def identity(self) -> Literal[0, 1] | None:
        return self._ufunc.identity

    @property
    @override
    def signature(self) -> str | None:
        return self._ufunc.signature


class UFuncWrapper(BaseUFuncWrapper[_T]):
    @property
    @override
    def types(self) -> list[str]:
        return self._ufunc.types  # pyright: ignore[reportReturnType]

    def resolve_dtypes(
        self,
        /,
        dtypes: tuple[np.dtype[Any] | type[Any] | None, ...],
        *,
        signature: tuple[np.dtype[Any] | None, ...] | None = None,
        casting: "_CastingKind | None" = None,
        reduction: bool = False,
    ) -> tuple[np.dtype[Any], ...]:
        return self._ufunc.resolve_dtypes(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
            dtypes,
            signature=signature,
            casting=casting,
            reduction=reduction,
        )


def can_cache(
    wrapped: Callable[..., Any],
    *,
    cache: bool | None = None,
) -> bool:
    if cache is False:
        return False
    filename = inspect.getfile(wrapped)
    if wrapped.__closure__:
        if not cache:
            return False
        warnings.warn(
            "Non-local variables captured as constants",
            kn.CacheWarning,
            stacklevel=4,
        )
    if filename not in linecache.cache:
        if not cache:
            return False
        warnings.warn(
            f"Cannot locate the source of {wrapped}",
            kn.CacheWarning,
            stacklevel=4,
        )
    return True


# ruff: disable[builtin-argument-shadowing]
def _init_error_details(
    sig: inspect.Signature,
    type: str,
    loc: tuple[int | str, ...],
    msg: str,
    input: object,
    **_: "Unused",
) -> pydantic_core.InitErrorDetails:
    match loc:
        case int() as i, *rest:
            it = itertools.islice(sig.parameters, i, None)
            loc = next(it), *rest
        case _:
            pass
    return {
        # https://github.com/pydantic/pydantic-core/issues/963
        "type": pydantic_core.PydanticCustomError(
            cast("LiteralString", type),
            cast("LiteralString", msg),
        ),
        "loc": loc,
        "input": input,
    }
# ruff: enable[builtin-argument-shadowing]


def _with_parameter_names(
    wrapped: Callable[..., object],
    sig: inspect.Signature,
) -> Callable[..., object]:
    # Temporary fix for https://github.com/pydantic/pydantic/issues/6791
    def wrapper(*args: object) -> object:
        try:
            return wrapped(*args)
        except pydantic.ValidationError as e:
            raise e.from_exception_data(
                e.title,
                line_errors=[
                    _init_error_details(sig, **detail) for detail in e.errors()
                ],
            ) from None
    return wrapper
