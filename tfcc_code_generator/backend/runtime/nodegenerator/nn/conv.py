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
from ...proto.operations import nn_pb2


class Conv(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.nn.Conv)

    @property
    def operation(self):
        if len(self.inputs[0].shape) == 4:
            operation = nn_pb2.Conv2D()
            operation.padding_height = self.node.padding_height
            operation.padding_width = self.node.padding_width
            operation.stride_height = self.node.stride_height
            operation.stride_width = self.node.stride_width
            operation.dilate_height = self.node.dilate_height
            operation.dilate_width = self.node.dilate_width
            operation.group = self.node.group
            operation.nchw = self.node.nchw
        else:
            raise RuntimeError("Unsupport conv type")

        return operation
