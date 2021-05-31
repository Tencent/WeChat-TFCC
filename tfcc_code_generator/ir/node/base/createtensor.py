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


class CreateTensor(Node):
    def update_attributes(self, value, dtype):
        assert isinstance(value, np.ndarray)
        assert value.size == 1
        self._value = value
        self._dtype = ir.framework.DataType(dtype)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_vector() and isinstance(self.inputs[0].shape[0], int)

        if isinstance(self.inputs[0].incomplete_data, list):
            shape = self.inputs[0].incomplete_data
        else:
            shape = [
                self.create_shape_name("create_tensor")
                for _ in range(self.inputs[0].shape[0])
            ]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.dtype
        self.outputs[0].shape = shape

    @property
    def value(self):
        return self._value

    @property
    def dtype(self):
        return self._dtype

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        data = np.ones(self.inputs[0].data.tolist())
        data = data * self.value
        self.outputs[0].shape = list(self.inputs[0].data.tolist())
        self.outputs[0].data = data.astype(self.dtype.numpy_dtype)

    @property
    def attributes(self):
        return {"dtype": self.dtype, "value": self.value.tolist()}
