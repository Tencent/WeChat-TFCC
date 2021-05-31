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


class Concat(Node):
    def update_attributes(self, axis: dict):
        assert isinstance(axis, int) and axis >= 0
        self._axis = axis

    def inference(self):
        assert len(self.inputs) > 1
        assert len(self.outputs) == 1
        assert all([x.is_tensor() for x in self.inputs])
        assert len(set([len(x.shape) for x in self.inputs])) == 1
        assert len(set([x.dtype for x in self.inputs])) == 1
        assert self.axis < len(self.inputs[0].shape)

        shape = []
        for i in range(len(self.inputs[0].shape)):
            s = self.inputs[0].shape[i]
            for x in self.inputs:
                if isinstance(s, int):
                    break
                if isinstance(s, str) and isinstance(x.shape[i], int):
                    s = x.shape[i]
            shape.append(s)

        dim = 0
        for x in self.inputs:
            if isinstance(x.shape[self.axis], int):
                dim += x.shape[self.axis]
            else:
                dim = self.create_shape_name("concat")
                break
        shape[self.axis] = dim

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def axis(self):
        return self._axis

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        self.outputs[0].data = np.concatenate(
            [symbol.data for symbol in self.inputs], axis=self.axis
        )

    @property
    def attributes(self):
        return {"axis": self.axis}
