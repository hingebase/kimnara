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

__all__ = ["call", "locate"]

import ast
import ctypes
import functools
import sys
from typing import no_type_check

import executing
from numba.core import types  # pyright: ignore[reportMissingTypeStubs]
from numba.core.typing import (  # pyright: ignore[reportMissingTypeStubs]
    ctypes_utils,
    typeof,
)
from typing_extensions import Any, TypeVar

import kimnara as kn
from kimnara import _spec, _utils

if sys.version_info >= (3, 14):
    try:
        from ctypes import c_double_complex, c_float_complex

        from numba.core.typing.ctypes_utils import (  # pyright: ignore[reportMissingTypeStubs]
            _FROM_CTYPES,  # ruff: ignore[import-private-name]
            _TO_CTYPES,  # ruff: ignore[import-private-name]
        )
    except ImportError:
        pass
    else:
        _FROM_CTYPES[c_float_complex] = types.complex64
        _FROM_CTYPES[c_double_complex] = types.complex128
        _TO_CTYPES[types.complex64] = c_float_complex
        _TO_CTYPES[types.complex128] = c_double_complex

_Inferable = ast.AnnAssign | ast.Expr  # | ast.Return
_T = TypeVar("_T", bound="ctypes._CData")  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
_Target = ast.Attribute | ast.Name


def call(
    op: executing.executing.Executing,
    args: tuple[object, ...],
    func_spec: int | tuple[str, ctypes.CDLL],
) -> object:
    restype = _restype(op)
    cargs, cargtypes = zip(*map(_to_ctypes, args), strict=True)
    cfunc = ctypes.CFUNCTYPE(restype, *cargtypes)(func_spec)  # pyright: ignore[reportUnknownArgumentType]
    return cfunc(*cargs)


def locate() -> executing.executing.Executing:
    op = executing.Source.executing(sys._getframe(2))  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[private-member-access]
    node = op.node
    if not isinstance(node, ast.Call):
        message = "Cannot locate the caller"
        if node:
            message = f"{message}:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    if not isinstance(node.parent, _Inferable):
        message = f"Cannot infer the return type:\n{ast.unparse(node.parent)}"
        raise kn.TypeInferenceError(message)
    return op


@functools.singledispatch
def _expr(node: ast.AST) -> str | ast.Expression:
    message = f"Invalid annotation:\n{ast.unparse(node)}"
    raise kn.TypeInferenceError(message)


@_expr.register(ast.Attribute)
@_expr.register(ast.Name)
@_expr.register(ast.Subscript)
def _(node: _Target | ast.Subscript) -> ast.Expression:
    return ast.Expression(node)


@_expr.register
def _(node: ast.Constant) -> str:
    value = node.value
    if not isinstance(value, str):
        message = f"Invalid annotation:\n{ast.unparse(node)}"
        raise kn.TypeInferenceError(message)
    return value


def _restype(
    op: executing.executing.Executing,
) -> type["ctypes._CData"] | None:  # pyright: ignore[reportPrivateUsage]
    node = op.node.parent
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, _Target):
            message = f"Invalid statement:\n{ast.unparse(node)}"
            raise kn.TypeInferenceError(message)
        expr = _expr(node.annotation)
        code = compile(expr, op.source.filename, "eval")
        ann = eval(code, op.frame.f_globals)  # ruff: ignore[suspicious-eval-usage]
        ctx = _spec.TypingContext(align=kn.A, allow_scalar=True)
        return ctx.infer(ann).to_ctypes()
    return None


@functools.singledispatch
@no_type_check
def _to_ctypes(arg: object) -> tuple[Any, type[Any]]:
    try:
        # Make the type conversion consistent in ctypes and Numba
        argtype = typeof.typeof(arg)
    except ValueError:
        pass
    else:
        try:
            cargtype = ctypes_utils.to_ctypes(argtype)
        except TypeError:
            pass
        else:
            return arg, cargtype
    try:
        wrapped = arg._as_parameter_
    except AttributeError:
        pass
    else:
        return _to_ctypes(wrapped)
    message = f"Cannot convert {_utils.base_repr(arg)} to ctypes"
    raise kn.TypeInferenceError(message)


@_to_ctypes.register
def _(_: None) -> tuple[None, type[ctypes.c_void_p]]:
    return None, ctypes.c_void_p


@_to_ctypes.register(ctypes.Array.__mro__[-2])
def _(carg: _T) -> tuple[_T, type[_T]]:
    return carg, type(carg)
