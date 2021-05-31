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


class MaxPool(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.nn.MaxPool)

    @property
    def operation(self):
        if len(self.inputs[0].shape) == 4:
            operation = nn_pb2.MaxPool2D()
            operation.kernel_height = self.node.kernels[0]
            operation.kernel_width = self.node.kernels[1]
            operation.padding_height = self.node.pads[0]
            operation.padding_width = self.node.pads[1]
            operation.stride_height = self.node.strides[0]
            operation.stride_width = self.node.strides[1]
            operation.nchw = self.node.data_format == self.node.DataFormat.NCHW
        elif len(self.inputs[0].shape) == 3:
            operation = nn_pb2.MaxPool1D()
            operation.kernel = self.node.kernels[0]
            operation.padding = self.node.pads[0]
            operation.stride = self.node.strides[0]
            operation.ncw = self.node.data_format == self.node.DataFormat.NCW
        else:
            raise RuntimeError("Unsupport max pool type")

        return operation
