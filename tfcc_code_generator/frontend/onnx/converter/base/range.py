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


class Range(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        start_to_value_name = graph.context.create_symbol_name(node_proto.input[0])
        limit_to_value_name = graph.context.create_symbol_name(node_proto.input[1])
        delta_to_value_name = graph.context.create_symbol_name(node_proto.input[2])
        output_name = graph.context.create_symbol_name(node_proto.output[0])
        graph.append_node(
            ir.node.base.ToValue(
                node_proto.name + ":0",
                graph,
                [node_proto.input[0]],
                [start_to_value_name],
            )
        )
        graph.append_node(
            ir.node.base.ToValue(
                node_proto.name + ":1",
                graph,
                [node_proto.input[1]],
                [limit_to_value_name],
            )
        )
        graph.append_node(
            ir.node.base.ToValue(
                node_proto.name + ":2",
                graph,
                [node_proto.input[2]],
                [delta_to_value_name],
            )
        )
        graph.append_node(
            ir.node.base.Range(
                node_proto.name + ":3",
                graph,
                [start_to_value_name, limit_to_value_name, delta_to_value_name],
                [output_name],
            )
        )
        graph.append_node(
            ir.node.base.ToTensor(
                node_proto.name + ":4", graph, [output_name], node_proto.output
            )
        )

        return True

    @property
    def accept_versions(self) -> set:
        # TODO support Reshape-1
        return set([11])
