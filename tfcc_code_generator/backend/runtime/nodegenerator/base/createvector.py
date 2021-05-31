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


class CreateVector(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.CreateVector)

    @property
    def operation(self):
        operation = base_pb2.CreateVector()
        values = []
        for v in self.node.vec:
            if isinstance(v, list):
                for x in v:
                    value_proto = base_pb2.CreateVector.Value()
                    if isinstance(x, int):
                        value_proto.int64_value = x
                    elif isinstance(x, float):
                        value_proto.double_value = x
                    else:
                        raise RuntimeError("vec type error")
                    values.append(value_proto)
            elif isinstance(v, int):
                value_proto = base_pb2.CreateVector.Value()
                value_proto.pos = v
                values.append(value_proto)
            else:
                raise RuntimeError("vec type error")
        operation.values.extend(values)
        operation.data_type = data_type_to_proto(self.node.dtype)

        return operation
