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


class SliceV2(Node):
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
        assert self.inputs[1].dtype == self.inputs[2].dtype
        assert self.axis < len(self.inputs[0].shape)

        shape = self.inputs[0].shape
        if (
            isinstance(self.inputs[0].shape[self.axis], int)
            and self.inputs[1].is_constant()
            and self.inputs[2].is_constant()
        ):
            start = self.inputs[1].data.tolist()[0]
            while start < 0:
                start += self.inputs[0].shape[self.axis]
            end = self.inputs[2].data.tolist()[0]
            while end < 0:
                end += self.inputs[0].shape[self.axis]
            end = min(self.inputs[0].shape[self.axis], end)
            assert start < end
            shape[self.axis] = end - start
        else:
            shape[self.axis] = self.create_shape_name("slice_v2")

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis}

    def calculate(self):
        start = self.inputs[1].data.tolist()[0]
        end = self.inputs[2].data.tolist()[0]
        data = np.reshape(self.inputs[0].data, self.inputs[0].shape)
        data = np.split(data, [start, end], self.axis)[1]
        data = data.astype(self.inputs[0].dtype.numpy_dtype)
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        self.outputs[0].data = data
