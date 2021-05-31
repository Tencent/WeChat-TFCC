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
import ir.framework
from ir.node import Node
from ir.common import get_broadcast_shape


class FusionOpDynamicShape(Node):
    def update_attributes(self, op_types):
        self._op_types = op_types
        assert isinstance(op_types, list)
        assert len(op_types) > 0

    def inference(self):
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0

        shape = get_broadcast_shape(
            [inp.shape for inp in self.inputs], self.graph.context
        )
        assert all([not inp.is_value() for inp in self.inputs])
        stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].stype = stype
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    def calculate(self):
        pass

    @property
    def op_types(self):
        return self._op_types

    @property
    def attributes(self):
        return {"op_types": self._op_types}
