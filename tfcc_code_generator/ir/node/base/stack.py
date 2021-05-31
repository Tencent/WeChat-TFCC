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


class Stack(Node):
    def update_attributes(self, axis):
        assert isinstance(axis, int) and axis >= 0
        self._axis = axis

    def inference(self):
        assert len(self.inputs) > 0
        assert len(self.outputs) == 1
        assert all([inp.is_tensor() for inp in self.inputs])
        assert len(set([inp.dtype for inp in self.inputs])) == 1
        shape = self.inputs[0].shape
        for inp in self.inputs:
            for i in range(len(shape)):
                assert (
                    shape[i] == inp.shape[i]
                    or isinstance(shape[i], str)
                    or isinstance(inp.shape[i], str)
                )
                if isinstance(shape[i], str) and isinstance(inp.shape[i], int):
                    shape[i] = inp.shape[i]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        shape.insert(self.axis, len(self.inputs))
        self.outputs[0].shape = shape

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis}

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        inp_data = [inp.data for inp in self.inputs]
        self.outputs[0].data = np.stack(inp_data, self.axis)
