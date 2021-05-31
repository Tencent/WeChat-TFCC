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

import onnx
import ir.node
import ir.framework
import frontend.onnx.common
from ..converter import Converter


class LogSoftmax(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        axis = 1
        if "axis" in attributes:
            axis = attributes["axis"]

        input_symbol = graph.get_symbol(node_proto.input[0])
        while axis < 0:
            axis = axis + len(input_symbol.shape)

        node_name = node_proto.name + ":0"
        softmaxOutputName = graph.context.create_symbol_name(node_proto.output[0])
        graph.append_node(
            ir.node.math.Softmax(
                node_name, graph, node_proto.input, [softmaxOutputName], axis=axis
            )
        )

        node_name = node_proto.name + ":1"
        graph.append_node(
            ir.node.math.Log(node_name, graph, [softmaxOutputName], node_proto.output)
        )
        return True

    @property
    def accept_versions(self):
        return set([1, 11])
