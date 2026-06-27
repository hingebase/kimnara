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

__all__ = ["Alignment", "Mutable", "scalar"]

import contextlib
import math
import operator
import sys

from optype.typing import AnyComplex
from typing_extensions import (
    NamedTuple,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
)

if sys.version_info >= (3, 11):
    _Complex = SupportsComplex
else:
    _Complex = SupportsComplex | complex


class Alignment(NamedTuple):
    name: str
    multiple_of: int = 1
    not_multiple_of: float = math.nan


class Mutable:
    pass


def scalar(value: AnyComplex) -> complex:
    if isinstance(value, SupportsIndex):
        with _suppress:
            return operator.index(value)
    if isinstance(value, SupportsFloat):
        with _suppress:
            return float(value)
    if isinstance(value, _Complex):
        with _suppress:
            return complex(value)
    class_repr = type.__repr__(type(value))  # noqa: PLC2801
    message = f"Cannot convert object of type {class_repr} to a numeric scalar"
    raise TypeError(message)


_suppress = contextlib.suppress(Exception)
