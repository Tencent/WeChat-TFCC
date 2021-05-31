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


class GetDimension(Node):
    def update_attributes(self, axis: int, dtype):
        assert isinstance(axis, int)
        assert axis >= 0
        self._axis = axis
        self._dtype = ir.framework.DataType(dtype)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1

        assert self.axis < len(self.inputs[0].shape)

        self.outputs[0].stype = ir.framework.SymbolType.VALUE
        self.outputs[0].dtype = self.dtype
        self.outputs[0].shape = [1]

    @property
    def axis(self):
        return self._axis

    @property
    def dtype(self):
        return self._dtype

    @property
    def attributes(self):
        return {"axis": self.axis, "dtype": self.dtype}

    def calculatable(self):
        return isinstance(self.inputs[0].shape[self.axis], int)

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        self.outputs[0].data = np.asarray(
            [self.inputs[0].shape[self.axis]], dtype=self.dtype.numpy_dtype
        )

    def calculate_incomplete_data(self):
        self.outputs[0].incomplete_data = [self.inputs[0].shape[self.axis]]
