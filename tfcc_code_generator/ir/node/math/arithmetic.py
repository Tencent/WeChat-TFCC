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

from ir.node import Node
import ir.framework
from ir.common import get_broadcast_shape
import numpy as np


class Arithmetic(Node):
    def compute(self, a: np.ndarray, b: np.ndarray):
        raise NotImplementedError

    def compute_incomplete_data(self, a: list, b: list):
        return None

    def inference(self):
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor() or self.inputs[0].is_value()
        assert self.inputs[1].is_tensor() or self.inputs[1].is_value()
        assert self.inputs[0].dtype == self.inputs[1].dtype

        shape = get_broadcast_shape(
            (self.inputs[0].shape, self.inputs[1].shape), self.graph.context
        )
        if self.inputs[0].is_value() and self.inputs[1].is_value():
            stype = ir.framework.SymbolType.VALUE
        else:
            stype = ir.framework.SymbolType.VARIABLE

        self.outputs[0].stype = stype
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        if self.outputs[0].is_tensor():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        elif self.outputs[0].is_value():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        else:
            raise RuntimeError("stype error")

        self.outputs[0].data = self.compute(self.inputs[0].data, self.inputs[1].data)

    def calculate_incomplete_data(self):
        if len(self.inputs[0].shape) != 1 or not isinstance(
            self.inputs[0].shape[0], int
        ):
            return
        if len(self.inputs[1].shape) != 1 or not isinstance(
            self.inputs[1].shape[0], int
        ):
            return
        if self.inputs[0].is_constant():
            data_1 = self.inputs[0].data.tolist()
        else:
            data_1 = self.inputs[0].incomplete_data
        if self.inputs[1].is_constant():
            data_2 = self.inputs[1].data.tolist()
        else:
            data_2 = self.inputs[1].incomplete_data

        if not data_1 or not data_2:
            return

        if len(data_1) == 1:
            data_1 = data_1 * len(data_2)
        if len(data_2) == 1:
            data_2 = data_2 * len(data_1)

        if len(data_1) != len(data_2):
            return

        data = self.compute_incomplete_data(data_1, data_2)
        if data:
            self.outputs[0].incomplete_data = data

    @property
    def attributes(self):
        return {}


class Add(Arithmetic):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

    def compute_incomplete_data(self, a: list, b: list):
        if len(a) != len(b):
            return None
        data = []
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if isinstance(x, str) and isinstance(y, str):
                return None
            if x == 0:
                data.append(y)
            elif y == 0:
                data.append(x)
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                data.append(x + y)
            else:
                return None
        return data


class Sub(Arithmetic):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a - b

    def compute_incomplete_data(self, a: list, b: list):
        if len(a) != len(b):
            return None
        data = []
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if isinstance(x, str) and isinstance(y, str):
                return None
            if y == 0:
                data.append(x)
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                data.append(x - y)
            else:
                return None
        return data


class Mul(Arithmetic):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def compute_incomplete_data(self, a: list, b: list):
        if len(a) != len(b):
            return None
        data = []
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if isinstance(x, str) and isinstance(y, str):
                return None
            if x == 1:
                data.append(y)
            elif y == 1:
                data.append(x)
            elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
                data.append(x * y)
            else:
                return None
        return data


class Div(Arithmetic):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a / b

    def compute_incomplete_data(self, a: list, b: list):
        if len(a) != len(b):
            return None
        data = []
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if isinstance(x, str) and isinstance(y, str):
                return None
            if y == 1:
                data.append(x)
            elif (
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and self.inputs[0].is_integer()
            ):
                data.append(x // y)
            elif (
                isinstance(x, (int, float))
                and isinstance(y, (int, float))
                and self.inputs[0].is_floating_point()
            ):
                data.append(x / y)
            else:
                return None
        return data
