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

import ir.node
from backend.runtime.nodegenerator.nodegenerator import NodeGenerator
from ...common import data_type_to_proto
from ...proto.operations import math_pb2


class ReduceMean(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.ReduceMean)

    @property
    def operation(self):
        operation = math_pb2.ReduceMean()
        operation.axes[:] = self.node.axes

        return operation


class ReduceSum(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.ReduceSum)

    @property
    def operation(self):
        operation = math_pb2.ReduceSum()
        operation.axes[:] = self.node.axes

        return operation


class ReduceProd(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.ReduceProd)

    @property
    def operation(self):
        operation = math_pb2.ReduceProd()
        operation.axes[:] = self.node.axes

        return operation


class ReduceMax(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.ReduceMax)

    @property
    def operation(self):
        operation = math_pb2.ReduceMax()
        operation.axes[:] = self.node.axes

        return operation


class ReduceMin(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.ReduceMin)

    @property
    def operation(self):
        operation = math_pb2.ReduceMin()
        operation.axes[:] = self.node.axes

        return operation
