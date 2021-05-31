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


class Expand(Node):
    def inference(self):
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert (
            self.inputs[1].is_vector()
            and isinstance(self.inputs[1].shape[0], int)
            and self.inputs[1].is_integer()
        )

        if self.inputs[1].is_constant():
            assert self.inputs[1].data.size >= len(self.inputs[0].shape)
            shape = [1] * (
                self.inputs[1].data.size - len(self.inputs[0].shape)
            ) + self.inputs[0].shape
            for i, s in enumerate(self.inputs[1].data.tolist()):
                s = int(s)
                assert s != 0
                if s > 0:
                    shape[i] = s
        else:
            shape = [
                self.create_shape_name("expand") for _ in range(self.inputs[1].shape[0])
            ]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def attributes(self):
        return {}

    def calculate(self):
        self.outputs[0].data = self.inputs[0].data * np.ones(
            self.inputs[1].data.tolist(), dtype=self.inputs[0].dtype.numpy_dtype
        )
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
