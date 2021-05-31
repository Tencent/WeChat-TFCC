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


class Split(Node):
    def update_attributes(self, axis: int):
        assert isinstance(axis, int) and axis >= 0
        self._axis = axis

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) > 0
        assert self.inputs[0].is_tensor()
        assert self.axis < len(self.inputs[0].shape)

        if isinstance(self.inputs[0].shape[self.axis], int):
            size = self.inputs[0].shape[self.axis] // len(self.outputs)
            assert size * len(self.outputs) == self.inputs[0].shape[self.axis]
        else:
            size = self.create_shape_name("split")
        shape = self.inputs[0].shape
        shape[self.axis] = size
        for out in self.outputs:
            out.stype = ir.framework.SymbolType.VARIABLE
            out.dtype = self.inputs[0].dtype
            out.shape = shape

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis}

    def calculate(self):
        for output, data in zip(
            self.outputs,
            np.split(self.inputs[0].data, len(self.output_names), axis=self.axis),
        ):
            output.data = data
            output.origin_stype = output.stype
            output.stype = ir.framework.SymbolType.CONSTANT_TENSOR
