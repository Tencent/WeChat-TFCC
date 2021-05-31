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


class Eye(Node):
    def update_attributes(self, dtype, k: int):
        assert isinstance(k, int)
        self._dtype = ir.framework.DataType(dtype)
        self._k = k

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_vector()

        assert isinstance(self.inputs[0].shape[0], str) or self.inputs[0].shape[0] == 2

        if self.inputs[0].is_constant():
            shape = [int(s) for s in self.inputs[0].data.tolist()]
        else:
            shape = [self.create_shape_name("eye") for _ in range(2)]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def k(self):
        return self._k

    @property
    def attributes(self):
        return {"dtype": self.dtype, "k": self.k}

    def calculate(self):
        data = np.eye(
            self.inputs[0].data[0],
            self.inputs[0].data[1],
            self.k,
            self.dtype.numpy_dtype,
        )
        self.outputs[0].data = data
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
