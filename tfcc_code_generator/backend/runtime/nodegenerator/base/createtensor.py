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
from ...proto.operations import base_pb2


class CreateTensor(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.CreateTensor)

    @property
    def operation(self):
        operation = base_pb2.CreateTensor()
        value = self.node.value.tolist()[0]
        if self.outputs[0].is_integer():
            operation.value.int64_value = value
        elif self.outputs[0].is_floating_point():
            operation.value.double_value = value
        else:
            raise RuntimeError("Unknow value")
        operation.data_type = data_type_to_proto(self.node.dtype)

        return operation
