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


class Cast(Node):
    def update_attributes(self, dtype):
        self._dtype = ir.framework.DataType(dtype)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor() or self.inputs[0].is_value()

        if self.inputs[0].is_tensor():
            self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        elif self.inputs[0].is_value():
            self.outputs[0].stype = ir.framework.SymbolType.VALUE
        else:
            raise RuntimeError("stype error")

        self.outputs[0].dtype = self.dtype
        self.outputs[0].shape = self.inputs[0].shape

    @property
    def dtype(self):
        return self._dtype

    def calculate(self):
        self.outputs[0].origin_stype = self.outputs[0].stype
        if self.inputs[0].is_tensor():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_TENSOR
        elif self.inputs[0].is_value():
            self.outputs[0].stype = ir.framework.SymbolType.CONSTANT_VALUE
        else:
            raise RuntimeError("stype error")
        self.outputs[0].data = self.inputs[0].data.astype(self.dtype.numpy_dtype)

    @property
    def attributes(self):
        return {"dtype": self._dtype}
