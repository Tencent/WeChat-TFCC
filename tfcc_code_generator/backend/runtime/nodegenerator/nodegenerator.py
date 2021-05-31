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

import ir.framework
from ..proto import model_pb2, common_pb2


class NodeGenerator(object):
    def __init__(self, node: ir.node.Node):
        self._node = node

    @classmethod
    def accept(cls, node: ir.node.Node):
        return False

    @property
    def node(self):
        return self._node

    @property
    def inputs(self):
        return self.node.inputs

    @property
    def outputs(self):
        return self.node.outputs

    @property
    def operation(self):
        raise NotImplementedError

    @property
    def proto(self) -> model_pb2.Node:
        result = model_pb2.Node()
        result.name = self.node.name
        result.inputs[:] = [symbol.name for symbol in self.inputs]
        result.outputs[:] = [symbol.name for symbol in self.outputs]

        operation = self.operation

        result.operation.Pack(operation)
        version = max(operation.VERSION.values())
        assert version > 0
        result.version = version

        return result
