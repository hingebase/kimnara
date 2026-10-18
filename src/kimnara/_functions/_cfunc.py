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

import ctypes
import functools
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import numba  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from numba.core import (  # pyright: ignore[reportMissingTypeStubs]
    cgutils,
    compiler,
    cpu,
    errors,
    imputils,
    registry,
    types,
)
from numba.core.typeconv.castgraph import (  # pyright: ignore[reportMissingTypeStubs]
    Conversion,
)
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    context,
    templates,
)
from numba.experimental.function_type import (  # pyright: ignore[reportMissingTypeStubs]
    lower_cast_dispatcher_to_function_type,  # pyright: ignore[reportUnknownVariableType]
)
from typing_extensions import TypeVar, override

import kimnara as kn
from kimnara import _spec, _utils
from kimnara._typing import CustomInliningRule, FastMathOptions

from . import _common

if TYPE_CHECKING:
    from ctypes import _CFunctionType  # pyright: ignore[reportPrivateUsage]

    from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]

NumbaT = TypeVar(
    "NumbaT",
    np.bool_,
    np.int8, np.int16, np.int32, np.int64, np.intp,
    np.uint8, np.uint16, np.uint32, np.uint64, np.uintp,
    np.float32, np.float64, np.complex64, np.complex128,
    bool, int, float, complex, None,
)
_Pointer = types.CPointer | types.ExternalFunctionPointer | types.RawPointer
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


class _CFunc(_common.Inferable[_T]):
    _address_or_callable: ClassVar[str]

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

    @functools.cached_property
    def _as_parameter_(self) -> "_CFunctionType":
        argtypes = [
            arg.to_ctypes() or _utils.unreachable()
            for arg in self.argtypes
        ]
        restype = self.restype.to_ctypes()
        ftype = ctypes.CFUNCTYPE(restype, *argtypes)
        return ftype(getattr(self, self._address_or_callable))

    def __call__(self, *args: object) -> _T:
        return self._as_parameter_(*args)


class NumbaCFunc(_CFunc[NumbaT], _common.Dispatchable):
    _address_or_callable = "address"
    _cres: compiler.CompileResult

    @override
    def __init__(
        self,
        wrapped: Callable[..., NumbaT],
        *,
        boundscheck: bool | None = None,
        cache: bool | None = None,
        error_model: Literal["python", "numpy"] = "numpy",
        fastmath: FastMathOptions = False,
        forceinline: bool = False,
        inline: Literal["always", "never"] | CustomInliningRule = "never",
        parallel: bool | str = False,
        pipeline_class: type[compiler.CompilerBase] | None = None,
    ) -> None:
        super().__init__(wrapped)
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = self.restype.to_numba()
        sig = restype(*argtypes)  # pyright: ignore[reportUnknownVariableType]
        decorator = numba.njit(  # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
            sig,
            boundscheck=boundscheck,
            cache=_common.can_cache(wrapped, cache=cache),
            error_model=error_model,
            fastmath=fastmath,  # pyright: ignore[reportArgumentType]
            forceinline=forceinline,
            inline=inline,
            no_cpython_wrapper=True,
            parallel=bool(parallel),
            pipeline_class=pipeline_class,
        )
        if parallel:
            with kn.threading.using_backend(parallel):
                func = cast("registry.CPUDispatcher", decorator(wrapped))
        else:
            func = cast("registry.CPUDispatcher", decorator(wrapped))
        self.dispatcher = func
        self._cres = func.get_compile_result(sig)  # pyright: ignore[reportUnknownMemberType]

    @property
    def _numba_type_(self) -> types.Dispatcher:
        return types.Dispatcher(self.dispatcher)

    @property
    def native_name(self) -> str:
        return self._cres.fndesc.llvm_cfunc_wrapper_name

    @property
    def address(self) -> int:
        match self._cres.library.get_pointer_to_function(self.native_name):
            case int(addr) if addr > 0:
                return addr
            case _:
                return _utils.unreachable()


class PyCFunc(_CFunc[PythonT], types.WrapperAddressProtocol):
    _address_or_callable = "__wrapped__"

    @override
    def signature(self) -> templates.Signature:  # pyright: ignore[reportIncompatibleMethodOverride]
        argtypes = [arg.to_numba() for arg in self.argtypes]
        restype = self.restype.to_numba()
        return restype(*argtypes)  # pyright: ignore[reportUnknownVariableType]

    @override
    def __wrapper_address__(self) -> int:  # pyright: ignore[reportIncompatibleMethodOverride]
        ptr = ctypes.cast(self._as_parameter_, ctypes.c_void_p)
        return ptr.value or _utils.unreachable()


def _can_convert_from(
    self: types.ExternalFunctionPointer,
    typingctx: context.Context,
    other: types.Type,
) -> Conversion | None:
    if conversion := _super_can_convert_from(self, typingctx, other):
        return conversion
    match other:
        case types.Dispatcher(dispatcher=registry.CPUDispatcher() as disp):
            if self.cconv is None and not disp.targetoptions.get(  # pyright: ignore[reportUnknownMemberType]
                "no_cfunc_wrapper",
                False,
            ):
                try:
                    disp.get_compile_result(self.sig)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                except errors.NumbaError:
                    pass
                else:
                    return Conversion.safe
        case types.ExternalFunctionPointer(sig=self.sig, cconv=self.cconv):  # pyright: ignore[reportUnknownMemberType]
            return Conversion.safe
        case types.FunctionType():
            other = other.get_precise()
            if self.sig == other.signature and self.cconv == other.cconv:  # pyright: ignore[reportUnknownMemberType]
                return Conversion.safe
        case _:
            pass
    return None


def _can_convert_to(
    self: types.NoneType,
    typingctx: context.Context,
    other: types.Type,
) -> Conversion | None:
    if conversion := _super_can_convert_to(self, typingctx, other):
        return conversion
    if isinstance(other, _Pointer):
        return Conversion.safe
    return None


@imputils.lower_cast(types.Dispatcher, types.ExternalFunctionPointer)  # pyright: ignore[reportUnknownMemberType]
def _(
    context: cpu.CPUContext,
    builder: "ir.IRBuilder",
    fromty: types.Dispatcher,
    toty: types.ExternalFunctionPointer,
    val: "ir.Value",
) -> "ir.Instruction":
    ftype = types.FunctionType(toty.sig)  # pyright: ignore[reportUnknownArgumentType]
    ftype.cconv = toty.cconv  # pyright: ignore[reportUnknownMemberType]
    val = lower_cast_dispatcher_to_function_type(  # pyright: ignore[reportUnknownVariableType]
        context, builder, fromty, ftype, val)
    return _lower_cast_function_type_to_function_pointer(
        context, builder, ftype, toty, val)  # pyright: ignore[reportUnknownArgumentType]


@imputils.lower_cast(  # pyright: ignore[reportUnknownMemberType]
    types.ExternalFunctionPointer,
    types.ExternalFunctionPointer,
)
def _(
    _context: cpu.CPUContext,
    _builder: "ir.IRBuilder",
    _fromty: types.ExternalFunctionPointer,
    _toty: types.ExternalFunctionPointer,
    val: "ir.Value",
) -> "ir.Value":
    return val


@imputils.lower_cast(types.FunctionType, types.ExternalFunctionPointer)  # pyright: ignore[reportUnknownMemberType]
def _lower_cast_function_type_to_function_pointer(
    context: cpu.CPUContext,
    builder: "ir.IRBuilder",
    fromty: types.FunctionType,
    toty: types.ExternalFunctionPointer,
    val: "ir.Value",
) -> "ir.Instruction":
    ptrty = cast("ir.Type", context.get_function_pointer_type(toty))  # pyright: ignore[reportUnknownMemberType]
    struct_cls = cast(
        "type[cgutils.ValueStructProxy]",
        cgutils.create_struct_proxy(fromty),  # pyright: ignore[reportUnknownMemberType]
    )
    struct = struct_cls(context, builder, value=val)
    return builder.bitcast(struct.c_addr, ptrty)  # pyright: ignore[reportReturnType, reportUnknownArgumentType, reportUnknownMemberType]


@imputils.lower_cast(types.NoneType, types.CPointer)  # pyright: ignore[reportUnknownMemberType]
@imputils.lower_cast(types.NoneType, types.ExternalFunctionPointer)  # pyright: ignore[reportUnknownMemberType]
@imputils.lower_cast(types.NoneType, types.RawPointer)  # pyright: ignore[reportUnknownMemberType]
def _(
    context: cpu.CPUContext,
    _builder: "ir.IRBuilder",
    _fromty: types.NoneType,
    toty: _Pointer,
    _val: "ir.Value",
) -> "ir.Value":
    return context.get_constant_null(toty)  # pyright: ignore[reportUnknownMemberType]


_super_can_convert_from = types.ExternalFunctionPointer.can_convert_from  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
_super_can_convert_to = types.NoneType.can_convert_to  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
types.ExternalFunctionPointer.can_convert_from = _can_convert_from  # pyright: ignore[reportAttributeAccessIssue]
types.NoneType.can_convert_to = _can_convert_to  # pyright: ignore[reportAttributeAccessIssue]
