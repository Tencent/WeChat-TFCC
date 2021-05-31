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

from numpy.lib.arraysetops import isin
from ir.node import Node
import ir.framework
from ir.common import get_broadcast_shape


class BatchNormalization(Node):
    def update_attributes(self, axis: int, epsilon: float):
        assert isinstance(axis, int) and axis >= 0
        assert isinstance(epsilon, float)
        self._axis = axis
        self._epsilon = epsilon

    # inputs: a, scale, offset, mean, var
    def inference(self):
        assert len(self.inputs) == 5
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert self.inputs[1].is_tensor()
        assert self.inputs[2].is_tensor()
        assert self.inputs[3].is_tensor()
        assert self.inputs[4].is_tensor()

        for v in self.inputs[1:]:
            assert len(v.shape) == 1 and v.shape[0] == self.inputs[0].shape[-1]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = self.inputs[0].shape

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis, "epsilon": self.epsilon}
