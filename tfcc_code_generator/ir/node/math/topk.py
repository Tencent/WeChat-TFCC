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


class TopK(Node):
    def inference(self):
        assert len(self.inputs) == 2
        assert len(self.outputs) == 2
        assert self.inputs[0].is_tensor()
        assert self.inputs[1].is_value() and self.inputs[1].is_integer()

        if self.inputs[1].is_constant():
            dim = self.inputs[1].data.tolist()[0]
        else:
            dim = self.create_shape_name("top_k")

        shape = self.inputs[0].shape[:-1] + [dim]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape
        self.outputs[1].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[1].dtype = ir.framework.DataType.INT64
        self.outputs[1].shape = shape

    @property
    def attributes(self):
        return {}
