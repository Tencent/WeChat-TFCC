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


class Range(Node):
    def inference(self):
        assert len(self.inputs) == 3
        assert len(self.outputs) == 1
        assert self.inputs[0].is_value()
        assert self.inputs[1].is_value()
        assert self.inputs[2].is_value()
        assert (
            self.inputs[0].dtype == self.inputs[1].dtype
            and self.inputs[1].dtype == self.inputs[2].dtype
        )

        shape = [self.create_shape_name("range")]

        self.outputs[0].stype = ir.framework.SymbolType.VECTOR
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def attributes(self):
        return {}

    def calculate(self):
        start = self.inputs[0].data.tolist()[0]
        limit = self.inputs[1].data.tolist()[0]
        delta = self.inputs[2].data.tolist()[0]

        data = np.arange(
            start=start, stop=limit, step=delta, dtype=self.outputs[0].dtype.numpy_dtype
        )
        self.outputs[0].shape = [data.size]
        self.outputs[0].data = data
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VECTOR
