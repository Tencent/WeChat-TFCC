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
from ir.common import get_broadcast_shape


class LocalResponseNormalization(Node):
    def update_attributes(
        self, axis: int, alpha: float, beta: float, bias: float, size: int
    ):
        assert isinstance(axis, int) and size >= 0
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert isinstance(bias, float)
        assert isinstance(size, int) and size > 0
        self._axis = axis
        self._alpha = alpha
        self._beta = beta
        self._bias = bias
        self._size = size

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.axis < len(self.inputs[0].shape)

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = self.inputs[0].shape

    @property
    def axis(self):
        return self._axis

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def bias(self):
        return self._bias

    @property
    def size(self):
        return self._size

    @property
    def attributes(self):
        return {
            "axis": self.axis,
            "alpha": self.alpha,
            "beta": self.beta,
            "bias": self.bias,
            "size": self.bias,
        }
