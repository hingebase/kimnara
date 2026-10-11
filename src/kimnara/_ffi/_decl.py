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

__all__ = ["Decl", "cydecl"]

import ctypes
import functools
import inspect
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, cast, no_type_check

import numba.extending  # pyright: ignore[reportMissingTypeStubs]
from numba.core.types import Type  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    context,
    templates,
)
from typing_extensions import Any, Protocol, Unpack

import kimnara as kn
from kimnara import _numba, _spec, _utils

if TYPE_CHECKING:
    from llvmlite import ir  # pyright: ignore[reportMissingTypeStubs]


class Decl(Protocol):
    def __call__(self, *args: object) -> Any: ...  # ruff: ignore[any-type]


def cydecl(
    wrapped: Callable[..., Any],
    module: types.ModuleType | str,
    function_name: str | None = None,
) -> Decl:
    module_name = cast("str", getattr(module, "__name__", module))
    if not function_name:
        function_name = wrapped.__name__
    address = cast(
        "int",
        numba.extending.get_cython_function_address(  # pyright: ignore[reportUnknownMemberType]
            module_name,
            function_name,
        ),
    )
    pysig = inspect.signature(wrapped, eval_str=True)
    argtypes, restype = _parse(pysig)
    sig = cast(
        "templates.Signature",
        restype.to_numba()(*[arg.to_numba() for arg in argtypes]),
    )
    cfunc = ctypes.CFUNCTYPE(
        restype.to_ctypes(),
        *[arg.to_ctypes() or _utils.unreachable() for arg in argtypes],
    )(address)
    typingctx = inspect.Parameter(
        "typingctx",
        inspect.Parameter.POSITIONAL_ONLY,
    )
    signature_holder: Any = types.SimpleNamespace(
        __signature__=pysig.replace(
            parameters=[typingctx, *pysig.parameters.values()],
        ),
    )

    @functools.wraps(wrapped)
    def wrapper(*args: object) -> object:
        return cfunc(*args)

    @numba.extending.intrinsic  # pyright: ignore[reportUnknownMemberType]
    @functools.wraps(signature_holder, assigned=())
    def intrinsic(
        _typingctx: context.Context,
        *_args: Type,
    ) -> "kn.typing.Intrinsic[Unpack[tuple[ir.Value, ...]]]":
        return (
            sig,
            lambda context, builder, signature, args: _numba.call(
                # TODO(): Retrieve the address at most once utilizing
                # C++11 local static variables

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
    return wrapper


def _parse(pysig: inspect.Signature) -> tuple[list[_spec.Type], _spec.Type]:
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
    return args, ctx.infer_with_diagnostics(pysig.return_annotation)
