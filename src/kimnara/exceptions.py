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

"""Exception definitions."""

__all__ = [
    "BackendUnavailableError",
    "Error",
    "PerformanceWarning",
    "ValidationError",
]

import os
import sys
import warnings


class Error(Exception):
    """Base class for known run-time exceptions in Kimnara."""


class BackendUnavailableError(Error):
    """Threading backends cannot be loaded."""


class PerformanceWarning(UserWarning, Error):  # noqa: N818
    """Performance-related issues that users should be aware of."""


class TypeInferenceError(Error):
    """Fail to infer runtime type from annotations."""


class ValidationError(FloatingPointError, Error):
    """Error raised by `kn.ops.invalid`.

    This class inherits from `FloatingPointError` to match the behavior
    of `np.errstate(invalid="raise")`.
    """


if (
    sys.platform.startswith("linux")
    and os.getenv("WSL_DISTRO_NAME")
    # https://docs.pytest.org/en/stable/example/simple.html#detect-if-running-from-within-a-pytest-run
    and not os.getenv("PYTEST_VERSION")
):
    # https://github.com/microsoft/WSL/issues/12813
    message = (
        "WSL may incur significant performance penalty in your computation. "
        "Consider switching to native Linux or Windows."
    )
    warnings.warn(message, PerformanceWarning, stacklevel=4)
