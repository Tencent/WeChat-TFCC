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


class Where(Node):
    def inference(self):
        assert len(self.inputs) == 3
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert self.inputs[0].dtype == ir.framework.DataType.BOOL
        assert self.inputs[1].is_tensor() or self.inputs[1].is_value()
        assert self.inputs[2].is_tensor() or self.inputs[2].is_value()
        assert not (self.inputs[1].is_value() and self.inputs[2].is_value())
        assert self.inputs[1].dtype == self.inputs[2].dtype

        shape = get_broadcast_shape(
            (self.inputs[1].shape, self.inputs[2].shape), self.graph.context
        )

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[1].dtype
        self.outputs[0].shape = shape

    @property
    def attributes(self):
        return {}
