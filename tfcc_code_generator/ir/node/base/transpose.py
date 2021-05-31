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


class Transpose(Node):
    def update_attributes(self, perm):
        assert all([isinstance(x, int) and x >= 0 for x in perm])
        assert sorted(perm) == list(range(len(perm)))
        self._perm = list(perm)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert len(self.perm) == len(self.inputs[0].shape)

        shape = []
        for x in self.perm:
            shape.append(self.inputs[0].shape[x])

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def perm(self):
        return self._perm

    @property
    def attributes(self):
        return {"perm": self.perm}

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        self.outputs[0].data = np.transpose(self.inputs[0].data, self.perm)
