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


class ToValue(Node):
    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert all([isinstance(s, str) or s == 1 for s in self.inputs[0].shape])

        self.outputs[0].stype = ir.framework.SymbolType.VALUE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = [1]

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        self.outputs[0].data = self.inputs[0].data

    def calculate_incomplete_data(self):
        if isinstance(self.inputs[0].incomplete_data, list):
            self.outputs[0].incomplete_data = self.inputs[0].incomplete_data

    @property
    def attributes(self):
        return {}
