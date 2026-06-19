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

"""Threading utilities."""

__all__ = [
    "BackendUnavailableError",
    "OpenMPNumPyVectorize",
    "TBBNumPyVectorize",
    "openmp_available",
    "register_custom_backend",
    "tbb_available",
    "using_backend",
]

from kimnara.exceptions import BackendUnavailableError

from ._custom import register_custom_backend, using_backend
from ._omppool import OpenMPNumPyVectorize, openmp_available
from ._tbbpool import TBBNumPyVectorize, tbb_available
