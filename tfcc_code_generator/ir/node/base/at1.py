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


class At1(Node):
    def update_attributes(self, idx):
        self._idx = idx
        assert idx >= 0
        assert isinstance(idx, int)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_vector()

        self.outputs[0].stype = ir.framework.SymbolType.VALUE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = [1]

    @property
    def idx(self):
        return self._idx

    def calculatable(self):
        if super().calculatable():
            return True
        if isinstance(self.inputs[0].incomplete_data, list) and isinstance(
            self.inputs[0].incomplete_data[self.idx], (int, float)
        ):
            return True
        return False

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        if self.inputs[0].is_constant():
            self.outputs[0].data = np.asarray(
                [self.inputs[0].data.tolist()[self.idx]],
                dtype=self.inputs[0].dtype.numpy_dtype,
            )
        elif isinstance(self.inputs[0].incomplete_data[self.idx], (int, float)):
            self.outputs[0].data = np.asarray(
                [self.inputs[0].incomplete_data[self.idx]],
                dtype=self.inputs[0].dtype.numpy_dtype,
            )
        else:
            raise RuntimeError("Unknow error")

    def calculate_incomplete_data(self):
        if isinstance(self.inputs[0].incomplete_data, list):
            self.outputs[0].incomplete_data = [self.inputs[0].incomplete_data[self.idx]]

    @property
    def attributes(self):
        return {"idx": self.idx}
