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


class If(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        to_value_output_name = graph.context.create_symbol_name(node_proto.input[0])
        graph.append_node(
            ir.node.base.ToValue(
                node_proto.name + ":0",
                graph,
                [node_proto.input[0]],
                [to_value_output_name],
            )
        )

        attributes = self.get_attributes(node_proto)
        op_set = graph.context.frontend["op_set"]
        from frontend.onnx.frontend import graph2ir

        if not "then_branch" in attributes:
            return False
        then_graph, then_name_map = graph2ir(
            graph.context.create_graph_name(attributes["then_branch"].name),
            attributes["then_branch"],
            graph.model,
            op_set,
            graph,
        )
        if not "else_branch" in attributes:
            return False
        else_graph, else_name_map = graph2ir(
            graph.context.create_graph_name(attributes["else_branch"].name),
            attributes["else_branch"],
            graph.model,
            op_set,
            graph,
        )

        then_name_map_reverse = {}
        for name in then_name_map:
            then_name_map_reverse[then_name_map[name]] = name
        else_name_map_reverse = {}
        for name in else_name_map:
            else_name_map_reverse[else_name_map[name]] = name
        inputs = [to_value_output_name]
        for name in then_graph.inputs:
            inputs.append(then_name_map_reverse[name])
        for name in else_graph.inputs:
            inputs.append(else_name_map_reverse[name])

        if_node = ir.node.base.If(
            node_proto.name + ":0",
            graph,
            inputs,
            node_proto.output,
            then_graph.name,
            len(then_graph.inputs),
            else_graph.name,
            len(else_graph.inputs),
        )
        graph.append_node(if_node)
        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 11])
