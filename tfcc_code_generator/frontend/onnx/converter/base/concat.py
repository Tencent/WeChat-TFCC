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


class Concat(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        # adapt for op set v1
        if self.since_version == 1 and "axis" not in attributes:
            attributes["axis"] = 1

        axis = attributes["axis"]

        input_symbol = graph.get_symbol(node_proto.input[0])
        while axis < 0:
            axis = axis + len(input_symbol.shape)

        inputs = []
        for inp in node_proto.input:
            if (
                "empty_symbol" in graph.context.frontend
                and inp in graph.context.frontend["empty_symbol"]
            ):
                continue
            inputs.append(inp)

        if len(inputs) > 1:
            graph.append_node(
                ir.node.base.Concat(
                    node_proto.name, graph, inputs, node_proto.output, axis=axis
                )
            )
        else:
            graph.append_node(
                ir.node.base.Identity(node_proto.name, graph, inputs, node_proto.output)
            )

        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 4, 11])
