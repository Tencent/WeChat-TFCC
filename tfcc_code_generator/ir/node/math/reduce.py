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


class Reduce(Node):
    def update_attributes(self, axes):
        assert all([isinstance(axis, int) and axis >= 0 for axis in axes])
        self._axes = axes

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert all([axis < len(self.inputs[0].shape) for axis in self.axes])

        shape = self.inputs[0].shape
        for axis in self.axes:
            shape[axis] = 1

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def axes(self):
        return self._axes

    @property
    def attributes(self):
        return {"axes": self.axes}


class ReduceMean(Reduce):
    pass


class ReduceSum(Reduce):
    pass


class ReduceProd(Reduce):
    pass


class ReduceMax(Reduce):
    pass


class ReduceMin(Reduce):
    pass


class ReduceAll(Reduce):
    pass
