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


class OneHot(Node):
    def update_attributes(self, dtype, depth: int, off_value, on_value):
        assert isinstance(depth, int) and depth >= 0
        self._dtype = ir.framework.DataType(dtype)
        self._depth = depth
        self._off_value = off_value
        self._on_value = on_value

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert self.inputs[0].is_integer()

        shape = self.inputs[0].shape + [self.depth]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.dtype
        self.outputs[0].shape = shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def depth(self):
        return self._depth

    @property
    def off_value(self):
        return self._off_value

    @property
    def on_value(self):
        return self._on_value

    @property
    def attributes(self):
        return {
            "dtype": self.dtype,
            "depth": self.depth,
            "off_value": self.off_value,
            "on_value": self.on_value,
        }
