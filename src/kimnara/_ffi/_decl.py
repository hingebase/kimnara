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

__all__ = ["Decl", "cdecl", "cydecl"]

import ctypes
import functools
import inspect
import types
from collections.abc import Callable
from typing import cast, no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from llvmlite import binding, ir  # pyright: ignore[reportMissingTypeStubs]
from numba.core.types import Type  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing.context import (  # pyright: ignore[reportMissingTypeStubs]
    Context,
)
from typing_extensions import Any, Protocol, Unpack

import kimnara as kn
from kimnara import _numba, _spec, _utils

_Intrinsic = kn.typing.Intrinsic[Unpack[tuple[ir.Value, ...]]]
_MESSAGE = "Function {!r} has been defined with a different signature"


class Decl(Protocol):
    def __call__(self, *args: object) -> Any: ...  # ruff: ignore[any-type]


def cdecl(
    wrapped: Callable[..., Any],
    lib: str = "c",
    function_name: str | None = None,
) -> Decl:
    dll = ctypes.cdll[lib] if "c" != lib != "m" else _utils.libc
    if not function_name:
        function_name = wrapped.__name__
    pysig = inspect.signature(wrapped, eval_str=True)
    argtypes, restype = _parse(pysig)
    if cache := _cdecl_cache.get((dll._handle, function_name)):  # ruff: ignore[private-member-access]
        cached_argtypes, cached_restype, cached_decl = cache
        if cached_argtypes == argtypes and cached_restype == restype:
            return cached_decl
        raise kn.RedefinitionError(_MESSAGE.format(function_name))
    if "c" != lib != "m":
        binding.load_library_permanently(lib)  # pyright: ignore[reportUnknownMemberType]
    sig = restype.to_numba()(*[arg.to_numba() for arg in argtypes])  # pyright: ignore[reportUnknownVariableType]
    cfunc = ctypes.CFUNCTYPE(
        restype.to_ctypes(),
        *[arg.to_ctypes() or _utils.unreachable() for arg in argtypes],
    )((function_name, dll))
    signature_holder: Any = types.SimpleNamespace(
        __signature__=pysig.replace(
            parameters=[_typingctx, *pysig.parameters.values()],
        ),
    )

    @functools.wraps(wrapped)
    def wrapper(*args: object) -> object:
        return cfunc(*args)

    @numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
    @functools.wraps(signature_holder, assigned=())
    def intrinsic(_typingctx: Context, *_args: Type) -> _Intrinsic:
        return sig, lambda context, builder, signature, args: _numba.call(  # pyright: ignore[reportUnknownVariableType]
            function_name, context, builder, signature, args)

    @numba.extending.overload(wrapper, strict=False)  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    @no_type_check
    def _(*args: Type) -> object:
        if len(args) == len(sig.args):
            return lambda *args: intrinsic(*args)  # ruff: ignore[unnecessary-lambda]
        return None

    wrapper._as_parameter_ = cfunc  # pyright: ignore[reportAttributeAccessIssue]
    _cdecl_cache[dll._handle, function_name] = argtypes, restype, wrapper  # ruff: ignore[private-member-access]
    return wrapper


def cydecl(
    wrapped: Callable[..., Any],
    module: types.ModuleType | str,
    function_name: str | None = None,
) -> Decl:
    module_name = cast("str", getattr(module, "__name__", module))
    if not function_name:
        function_name = wrapped.__name__
    pysig = inspect.signature(wrapped, eval_str=True)
    argtypes, restype = _parse(pysig)
    if cache := _cydecl_cache.get((module_name, function_name)):
        cached_argtypes, cached_restype, cached_decl = cache
        if cached_argtypes == argtypes and cached_restype == restype:
            return cached_decl
        raise kn.RedefinitionError(_MESSAGE.format(function_name))
    address = cast(
        "int",
        numba.extending.get_cython_function_address(  # pyright: ignore[reportUnknownMemberType]
            module_name,
            function_name,
        ),
    )
    sig = restype.to_numba()(*[arg.to_numba() for arg in argtypes])  # pyright: ignore[reportUnknownVariableType]
    cfunc = ctypes.CFUNCTYPE(
        restype.to_ctypes(),
        *[arg.to_ctypes() or _utils.unreachable() for arg in argtypes],
    )(address)
    signature_holder: Any = types.SimpleNamespace(
        __signature__=pysig.replace(
            parameters=[_typingctx, *pysig.parameters.values()],
        ),
    )

    @functools.wraps(wrapped)
    def wrapper(*args: object) -> object:
        return cfunc(*args)

    @numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
    @functools.wraps(signature_holder, assigned=())
    def intrinsic(_typingctx: Context, *_args: Type) -> _Intrinsic:
        return (  # pyright: ignore[reportUnknownVariableType]
            sig,
            lambda context, builder, signature, args: _numba.call(
                # TODO(): Retrieve the address at most once utilizing
                # C++11 local static variables

                # cppyy.numba_ext cannot be used since it exposes C++
                # functions as uncacheable pointers

                # Before https://github.com/numba/numba/issues/7371
                # we may need to ship pre-compiled LLVM IR like
                # https://github.com/numba/numba/issues/2795#issuecomment-472013539
                _numba.get_cython_function_address(
                    module_name,
                    function_name,
                    context,
                    builder,
                ),

                context,
                builder,
                signature,
                args,
            ),
        )

    @numba.extending.overload(wrapper, strict=False)  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]
    @no_type_check
    def _(*args: Type) -> object:
        if len(args) == len(sig.args):
            return lambda *args: intrinsic(*args)  # ruff: ignore[unnecessary-lambda]
        return None

    wrapper._as_parameter_ = cfunc  # pyright: ignore[reportAttributeAccessIssue]
    _cydecl_cache[module_name, function_name] = argtypes, restype, wrapper
    return wrapper


def _parse(
    pysig: inspect.Signature,
) -> tuple[tuple[_spec.Type, ...], _spec.Type]:
    ctx = _spec.TypingContext(align=kn.A, allow_scalar=True)
    args: list[_spec.Type] = []
    for v in pysig.parameters.values():
        if v.POSITIONAL_ONLY is not v.kind is not v.POSITIONAL_OR_KEYWORD:
            message = "kwargs and varargs are disallowed"
            raise kn.TypeInferenceError(message)
        if v.default is not v.empty:
            message = "default parameter values are disallowed"
            raise kn.TypeInferenceError(message)
        arg = ctx.infer_with_diagnostics(
            v.annotation,
            "Failed to infer the type of argument {0.name!r}: {1}",
            v,
        )
        args.append(arg)
    ctx = ctx._replace(allow_none=True)
    return tuple(args), ctx.infer_with_diagnostics(pysig.return_annotation)


_cdecl_cache: dict[
    tuple[int, str],
    tuple[tuple[_spec.Type, ...], _spec.Type, Decl],
] = {}
_cydecl_cache: dict[
    tuple[str, str],
    tuple[tuple[_spec.Type, ...], _spec.Type, Decl],
] = {}
_typingctx = inspect.Parameter("typingctx", inspect.Parameter.POSITIONAL_ONLY)
