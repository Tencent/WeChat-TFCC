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


class Unsqueeze(Node):
    def update_attributes(self, axes):
        assert all([axis >= 0 for axis in axes]) and len(axes) > 0
        axes = axes.copy()
        axes = list(set(axes))
        self._axes = sorted(axes)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()

        shape = []
        pos = 0
        for i in range(len(self.axes) + len(self.inputs[0].shape)):
            if i in self.axes:
                shape.append(1)
            else:
                shape.append(self.inputs[0].shape[pos])
                pos += 1
        assert len(self.inputs[0].shape) + len(self.axes) == len(shape)

        self.outputs[0].stype = ir.framework.SymbolType.VIEW
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def axes(self):
        return self._axes

    @property
    def attributes(self):
        return {"axes": self.axes}

    def calculate(self):
        self.outputs[0].data = self.inputs[0].data.reshape(self.outputs[0].shape)
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
