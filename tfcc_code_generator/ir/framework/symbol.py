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

import numpy as np
from .types import DataType, SymbolType


class Symbol(object):
    def __init__(self, name):
        self._name = name
        self._dtype = None
        self._stype = None
        self._origin_stype = None
        self._shape = None
        self._data = None
        self._incomplete_data = None

    def __str__(self):
        return "name:{} dtype:{} stype:{} shape: {}".format(
            self._name, self._dtype, self._stype, self._shape
        )

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = DataType(value)

    @property
    def stype(self):
        return self._stype

    @stype.setter
    def stype(self, value):
        self._stype = SymbolType(value)

    @property
    def origin_stype(self):
        return self._origin_stype

    @origin_stype.setter
    def origin_stype(self, value):
        self._origin_stype = SymbolType(value)

    @property
    def shape(self):
        return self._shape.copy()

    @shape.setter
    def shape(self, value: list):
        value = list(value)
        for s in value:
            assert isinstance(s, (int, str))
        self._shape = value.copy()

    @property
    def data(self) -> np.ndarray:
        return self._data.copy()

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    @property
    def incomplete_data(self) -> list:
        return self._incomplete_data

    @incomplete_data.setter
    def incomplete_data(self, value: list):
        assert all([isinstance(v, (int, float, str)) for v in value])
        self._incomplete_data = list(value).copy()

    def is_constant(self):
        return (
            self.stype == SymbolType.CONSTANT_TENSOR
            or self.stype == SymbolType.CONSTANT_VALUE
            or self.stype == SymbolType.CONSTANT_VECTOR
        )

    def is_tensor(self):
        return (
            self.stype == SymbolType.VARIABLE
            or self.stype == SymbolType.VIEW
            or self.stype == self.stype.CONSTANT_TENSOR
        )

    def is_value(self):
        return self.stype == SymbolType.VALUE or self.stype == SymbolType.CONSTANT_VALUE

    def is_vector(self):
        return (
            self.stype == SymbolType.VECTOR or self.stype == SymbolType.CONSTANT_VECTOR
        )

    def is_integer(self):
        if self.dtype == DataType.INT8 or self.dtype == DataType.UINT8:
            return True
        if self.dtype == DataType.INT16 or self.dtype == DataType.UINT16:
            return True
        if self.dtype == DataType.INT32 or self.dtype == DataType.UINT32:
            return True
        if self.dtype == DataType.INT64 or self.dtype == DataType.UINT64:
            return True
        return False

    def is_signed(self):
        if self.dtype == DataType.INT8:
            return True
        if self.dtype == DataType.INT16:
            return True
        if self.dtype == DataType.INT32:
            return True
        if self.dtype == DataType.INT64:
            return True
        if self.dtype == DataType.FLOAT or self.dtype == DataType.DOUBLE:
            return True
        return False

    def is_floating_point(self):
        return self.dtype == DataType.FLOAT or self.dtype == DataType.DOUBLE

    def is_complex(self):
        return self.dtype == DataType.COMPLEX64

    def verify(self):
        if not isinstance(self.stype, SymbolType) or not isinstance(
            self.dtype, DataType
        ):
            return False
        if not self.shape:
            return False
        if self.is_constant():
            if self._data is None:
                return False
            if not all([isinstance(s, int) for s in self.shape]):
                return False
        if self.is_vector():
            if len(self.shape) != 1:
                return False
        if self.is_value():
            if self.shape != [1]:
                return False
        area = 1
        for s in self.shape:
            if isinstance(s, int):
                if s <= 0:
                    return False
                area *= s
            else:
                area = None
                break
        if self.incomplete_data is not None and not isinstance(
            self.incomplete_data, list
        ):
            return False
        if isinstance(self.incomplete_data, list):
            if all([isinstance(v, int) for v in self.incomplete_data]):
                return False
        if area is not None and isinstance(self.incomplete_data, list):
            if area != len(self.incomplete_data):
                return False
        if self.is_constant() and self.origin_stype is None:
            return False
        if not self.is_constant() and self.origin_stype is not None:
            return False
        return True

    def copy_to(self, dst_symbol):
        assert isinstance(dst_symbol, Symbol)
        dst_symbol.dtype = self.dtype
        dst_symbol.stype = self.stype
        dst_symbol.shape = self.shape
        if self._origin_stype is not None:
            dst_symbol.origin_stype = self.origin_stype
        if self._data is not None:
            dst_symbol.data = self.data
        if self._incomplete_data is not None:
            dst_symbol.incomplete_data = self.incomplete_data
        assert dst_symbol.verify()
