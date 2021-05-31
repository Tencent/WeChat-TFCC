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
from ir.node import Node
import ir.framework
from ir.common import get_broadcast_shape


class Relation(Node):
    def compute(self, a: np.ndarray, b: np.ndarray):
        raise NotImplementedError

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
        self.outputs[0].dtype = ir.framework.DataType.BOOL
        self.outputs[0].shape = shape

    @property
    def attributes(self):
        return {}

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        if self.outputs[0].is_tensor():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        elif self.outputs[0].is_value():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        else:
            raise RuntimeError("stype error")

        data = self.compute(self.inputs[0].data, self.inputs[1].data)
        data = data.astype(dtype=np.uint8)
        self.outputs[0].data = data


class Equal(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a == b


class UnEqual(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a != b


class Less(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a < b


class LessOrEqual(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a <= b


class Greater(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a > b


class GreaterOrEqual(Relation):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a >= b
