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


class Pad(Node):
    def update_attributes(self, axis: int):
        assert isinstance(axis, int) and axis >= 0
        self._axis = axis

    def inference(self):
        assert len(self.inputs) == 3
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert self.inputs[1].is_value()
        assert self.inputs[1].is_integer()
        assert self.inputs[2].is_value()
        assert self.inputs[2].is_integer()
        assert self.axis < len(self.inputs[0].shape)

        shape = self.inputs[0].shape
        if (
            isinstance(self.inputs[0].shape[self.axis], int)
            and self.inputs[1].is_constant()
            and self.inputs[2].is_constant()
        ):
            shape[self.axis] += int(
                self.inputs[1].data.tolist()[0] + self.inputs[2].data.tolist()[0]
            )
        else:
            shape[self.axis] = self.create_shape_name("pad")

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis}
