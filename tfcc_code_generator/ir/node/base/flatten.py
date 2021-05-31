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


class Flatten(Node):
    def update_attributes(self, axis: int):
        assert axis >= 0
        self._axis = axis

    def inference(self):
        assert len(self.inputs) == 1
        assert self.inputs[0].is_tensor()
        assert self.axis <= len(self.inputs[0].shape)

        s0 = 1
        for s in self.inputs[0].shape[: self.axis]:
            if isinstance(s, str):
                s0 = self.create_shape_name("flatten_s0")
                break
            else:
                s0 *= s

        s1 = 1
        for s in self.inputs[0].shape[self.axis :]:
            if isinstance(s, str):
                s1 = self.create_shape_name("flatten_s1")
                break
            else:
                s1 *= s

        self.outputs[0].stype = ir.framework.SymbolType.VIEW
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = [s0, s1]

    @property
    def axis(self):
        return self._axis

    @property
    def attributes(self):
        return {"axis": self.axis}

    def calculate(self):
        # TODO remove info log
        pass
