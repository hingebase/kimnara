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
]

import abc
import collections
import inspect
from collections.abc import Callable
from typing import Generic, Literal, cast

import numpy as np
import pint.registry_helpers
import pydantic
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
from typing_extensions import Any, Protocol, TypeVar, Unpack, override

import kimnara as kn
from kimnara import _spec
from kimnara._typing import Casting, Input, Outputs, UFuncKwargs

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
        key = value._sig  # pyright: ignore[reportPrivateUsage]
        cres = value._cache.load_overload(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType, reportUnknownVariableType]
            key,
            registry.cpu_target.target_context,
        )
        if cres is None:
            native_name = cast("str", value.native_name)
            fndesc = FunctionDescriptor.__new__(FunctionDescriptor)
            fndesc.mangled_name = native_name.removeprefix("cfunc.")
            cres = compiler.compile_result(  # pyright: ignore[reportUnknownMemberType]
                fndesc=fndesc,
                library=value._library,  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType, reportUnknownMemberType]
            )
        new.overloads = collections.OrderedDict([(key, cres)])  # pyright: ignore[reportUnknownArgumentType]
        new.py_func = value._pyfunc  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        # ruff: enable[private-member-access]

    def inspect_asm(self) -> str:
        [asm] = self._dispatcher.inspect_asm().values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return cast("str", asm)

    def inspect_cfg(self, **kwargs: object) -> _CFG:
        [cfg] = self._dispatcher.inspect_cfg(**kwargs).values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return cast("_CFG", cfg)

    def inspect_disasm_cfg(self) -> _DisasmCFG:
        [disasm_cfg] = self._dispatcher.inspect_disasm_cfg().values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return cast("_DisasmCFG", disasm_cfg)

    def inspect_llvm(self) -> str:
        [llvm] = self._dispatcher.inspect_llvm().values()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return cast("str", llvm)


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
            args.append(ctx.infer(v.annotation))
        ctx = self.output_context(align, pad_value)
        return tuple(args), ctx.infer(self.sig.return_annotation)

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


class Validator(Inferable[_T]):
    def validate_call(
        self,
        wrapped: Callable[..., object],
        sig: inspect.Signature | None = None,
    ) -> Callable[..., Any]:
        wrapper = cast(
            "Callable[..., object]",
            pint.registry_helpers.wraps(  # pyright: ignore[reportUnknownMemberType]
                _spec.ureg,  # pyright: ignore[reportArgumentType]
                ret=self.restype.to_units(),
                args=None,  # Handled by pydantic
            )(wrapped),
        )
        wrapper.__signature__ = sig = inspect.Signature(  # pyright: ignore[reportFunctionMemberAccess]
            parameters=[
                param.replace(annotation=arg.to_python())
                for arg, param in zip(
                    self.argtypes,
                    (sig or self.sig).parameters.values(),
                    strict=True,
                )
            ],
            return_annotation=self.restype.to_python(),
        )
        wrapper.__annotations__ = annotations = {
            k: v.annotation for k, v in sig.parameters.items()
        }
        annotations["return"] = sig.return_annotation
        return pydantic.validate_call(wrapper)


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
    @abc.abstractmethod
    def nin(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def nout(self) -> int:
        raise NotImplementedError

    @property
    def nargs(self) -> int:
        return self.nin + self.nout

    @property
    def ntypes(self) -> Literal[1]:
        return 1

    @property
    @abc.abstractmethod
    def types(self) -> list[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def identity(self) -> Literal[0, 1] | None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def signature(self) -> str | None:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args: Input, **kwargs: Unpack[UFuncKwargs]) -> Outputs:
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
        return cast("list[str]", self._ufunc.types)

    def resolve_dtypes(
        self,
        /,
        dtypes: tuple[np.dtype[Any] | type[Any] | None, ...],
        *,
        signature: tuple[np.dtype[Any] | None, ...] | None = None,
        casting: Casting | None = None,
        reduction: bool = False,
    ) -> tuple[np.dtype[Any], ...]:
        return self._ufunc.resolve_dtypes(
            dtypes,
            signature=signature,
            casting=casting,
            reduction=reduction,
        )
