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


class AveragePool(Node):
    def update_attributes(self, kernels, pads, strides):
        assert all([isinstance(v, int) for v in kernels])
        assert all([isinstance(v, int) for v in pads])
        assert all([isinstance(v, int) for v in strides])
        assert len(kernels) == len(pads) and len(pads) == len(strides)
        self._kernels = kernels
        self._pads = pads
        self._strides = strides

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert len(self.inputs[0].shape) == 4 or len(self.inputs[0].shape) == 3

        shape = []
        shape.append(self.inputs[0].shape[0])
        shape.append(self.inputs[0].shape[1])
        for i in range(len(self.inputs[0].shape) - 2):
            if isinstance(self.inputs[0].shape[i + 2], int):
                size = (
                    self.inputs[0].shape[i + 2] + 2 * self.pads[i] - self.kernels[i]
                ) // self.strides[i] + 1
            else:
                size = self.create_shape_name("average_pool")
            shape.append(size)

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def kernels(self):
        return self._kernels

    @property
    def pads(self):
        return self._pads

    @property
    def strides(self):
        return self._strides

    @property
    def attributes(self):
        return {"kernels": self.kernels, "pads": self.pads, "strides": self.strides}
