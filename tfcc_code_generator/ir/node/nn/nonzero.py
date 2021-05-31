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
from ir.common import get_broadcast_shape


class NonZero(Node):
    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()

        shape = [self.create_shape_name("non_zero"), len(self.inputs[0].shape)]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = ir.framework.DataType.INT64
        self.outputs[0].shape = shape

    def calculate(self):
        data = np.nonzero(np.reshape(self.inputs[0].data, self.inputs[0].shape))
        data = np.asarray(data, dtype=np.int64)
        data = data.T
        self.outputs[0].origin_stype = self.outputs[0].stype
        self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        self.outputs[0].data = data
        self.outputs[0].shape = list(data.shape)

    @property
    def attributes(self):
        return {}
