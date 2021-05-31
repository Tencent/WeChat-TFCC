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


class Reshape(Node):
    def inference(self):
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert (
            self.inputs[1].is_vector()
            and isinstance(self.inputs[1].shape[0], int)
            and self.inputs[1].is_integer()
        )

        area = None
        if all([isinstance(s, int) for s in self.inputs[0].shape]):
            area = 1
            for s in self.inputs[0].shape:
                area *= s

        if self.inputs[1].is_constant():
            shape = [
                int(x) if x > 0 else self.create_shape_name("reshape")
                for x in self.inputs[1].data.tolist()
            ]
        elif isinstance(self.inputs[1].incomplete_data, list):
            shape = [
                int(x)
                if isinstance(x, int) and x > 0
                else self.create_shape_name("reshape")
                for x in self.inputs[1].incomplete_data
            ]
        else:
            shape = [
                self.create_shape_name("reshape")
                for _ in range(self.inputs[1].shape[0])
            ]

        if sum([isinstance(s, str) for s in shape]) == 1 and isinstance(area, int):
            current_area = 1
            for s in shape:
                if isinstance(s, int):
                    current_area *= s
            for i, s in enumerate(shape):
                if isinstance(s, str):
                    assert area % current_area == 0
                    shape[i] = area // current_area
                    break

        if (
            sum([isinstance(s, str) for s in shape]) == 1
            and sum([isinstance(s, str) for s in self.inputs[0].shape]) == 1
        ):
            old_str_s = [s for s in self.inputs[0].shape if isinstance(s, str)][0]
            new_str_s = [s for s in shape if isinstance(s, str)][0]
            if [s for s in shape if not isinstance(s, str)] == [
                s for s in self.inputs[0].shape if not isinstance(s, str)
            ]:
                shape[shape.index(new_str_s)] = old_str_s

        self.outputs[0].stype = ir.framework.SymbolType.VIEW
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def attributes(self):
        return {}

    def calculate(self):
        self.outputs[0].data = self.inputs[0].data.reshape(self.outputs[0].shape)
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
