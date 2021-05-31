# Copyright 2021 Wechat Group, Tencent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import numpy as np


class SymbolType(enum.Enum):
    VARIABLE = 1
    VIEW = 2
    CONSTANT_TENSOR = 3
    VALUE = 4
    CONSTANT_VALUE = 5
    VECTOR = 6
    CONSTANT_VECTOR = 7


class DataType(enum.Enum):
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14

    @property
    def numpy_dtype(self):
        ctype_map = {
            DataType.FLOAT: np.dtype(np.float32),
            DataType.UINT8: np.dtype(np.uint8),
            DataType.INT8: np.dtype(np.int8),
            DataType.UINT16: np.dtype(np.uint16),
            DataType.INT16: np.dtype(np.int16),
            DataType.INT32: np.dtype(np.int32),
            DataType.INT64: np.dtype(np.int64),
            DataType.STRING: np.dtype(np.bytes_),
            DataType.BOOL: np.dtype(np.bool),
            DataType.DOUBLE: np.dtype(np.double),
            DataType.UINT32: np.dtype(np.uint32),
            DataType.UINT64: np.dtype(np.uint64),
            DataType.COMPLEX64: np.dtype(np.complex64),
        }
        return ctype_map[self]
