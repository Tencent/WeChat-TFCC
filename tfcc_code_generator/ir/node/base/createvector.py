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


class CreateVector(Node):
    def update_attributes(self, vec, dtype):
        assert isinstance(vec, list)
        assert all([isinstance(v, (int, list)) for v in vec])
        self._vec = vec
        self._dtype = ir.framework.DataType(dtype)

    def inference(self):
        assert len(self.outputs) == 1
        assert all([symbol.is_value() or symbol.is_vector() for symbol in self.inputs])

        dim = 0
        for v in self.vec:
            if isinstance(v, int) and isinstance(self.inputs[v].shape[0], int):
                dim += self.inputs[v].shape[0]
            elif isinstance(v, int) and isinstance(self.inputs[v].shape[0], str):
                dim = self.create_shape_name("create_vector")
                break
            elif isinstance(v, list):
                dim += len(v)
            else:
                raise RuntimeError("Unknow error")

        self.outputs[0].stype = ir.framework.SymbolType.VECTOR
        self.outputs[0].dtype = self.dtype
        self.outputs[0].shape = [dim]

    @property
    def vec(self):
        return self._vec

    @property
    def dtype(self):
        return self._dtype

    @property
    def attributes(self):
        return {"vec": self.vec, "dtype": self.dtype}

    def calculate(self):
        data = []
        for v in self.vec:
            if isinstance(v, int):
                data = data + self.inputs[v].data.tolist()
            elif isinstance(v, list):
                data += v
            else:
                raise RuntimeError("Unknow error")
        data = np.asarray(data, dtype=self.dtype.numpy_dtype)
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VECTOR
        self.outputs[0].data = data

    def calculate_incomplete_data(self):
        if not all(
            [
                isinstance(symbol.incomplete_data, list) or symbol.is_constant()
                for symbol in self.inputs
            ]
        ):
            return
        incomplete_data = []
        for v in self.vec:
            if isinstance(v, int):
                if self.inputs[v].is_constant():
                    incomplete_data = incomplete_data + list(
                        self.inputs[v].data.tolist()
                    )
                else:
                    incomplete_data = incomplete_data + self.inputs[v].incomplete_data
            elif isinstance(v, list):
                incomplete_data += v
            else:
                raise RuntimeError("Unknow error")
        self.outputs[0].incomplete_data = incomplete_data
