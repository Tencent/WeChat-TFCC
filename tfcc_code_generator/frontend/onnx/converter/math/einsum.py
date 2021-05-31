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


class Einsum(Converter):
    def process_bfnd_ndh_bfh(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        dim_b_name = graph.context.create_symbol_name(node_proto.input[0])
        graph.append_node(
            ir.node.base.GetDimension(
                node_proto.name + ":0",
                graph,
                [node_proto.input[0]],
                [dim_b_name],
                axis=0,
                dtype=ir.framework.DataType.INT64,
            )
        )
        dim_f_name = graph.context.create_symbol_name(node_proto.input[0])
        graph.append_node(
            ir.node.base.GetDimension(
                node_proto.name + ":1",
                graph,
                [node_proto.input[0]],
                [dim_f_name],
                axis=1,
                dtype=ir.framework.DataType.INT64,
            )
        )
        dim_h_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.GetDimension(
                node_proto.name + ":2",
                graph,
                [node_proto.input[1]],
                [dim_h_name],
                axis=2,
                dtype=ir.framework.DataType.INT64,
            )
        )

        a_shape_name = graph.context.create_symbol_name(node_proto.input[0])
        b_shape_name = graph.context.create_symbol_name(node_proto.input[1])

        graph.append_node(
            ir.node.base.CreateVector(
                node_proto.name + ":3",
                graph,
                [dim_b_name, dim_f_name],
                [a_shape_name],
                vec=[0, 1, [-1]],
                dtype=ir.framework.DataType.INT64,
            )
        )
        graph.append_node(
            ir.node.base.CreateVector(
                node_proto.name + ":4",
                graph,
                [dim_h_name],
                [b_shape_name],
                vec=[[-1], 0],
                dtype=ir.framework.DataType.INT64,
            )
        )

        a_name = graph.context.create_symbol_name(node_proto.input[0])
        b_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.Reshape(
                node_proto.name + ":5",
                graph,
                [node_proto.input[0], a_shape_name],
                [a_name],
            )
        )
        graph.append_node(
            ir.node.base.Reshape(
                node_proto.name + ":6",
                graph,
                [node_proto.input[1], b_shape_name],
                [b_name],
            )
        )
        graph.append_node(
            ir.node.math.Matmul(
                node_proto.name + ":7", graph, [a_name, b_name], node_proto.output
            )
        )
        return True

    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        equation = attributes["equation"].decode("utf8")
        a_pattern = equation.split("->")[0].split(",")[0]
        b_pattern = equation.split("->")[0].split(",")[1]
        c_pattern = equation.split("->")[1]

        if len(a_pattern) != 4 or len(b_pattern) != 3 or len(c_pattern) != 3:
            return False
        if a_pattern[2:] == b_pattern[:2] and c_pattern == (
            a_pattern[:2] + b_pattern[2:]
        ):
            return self.process_bfnd_ndh_bfh(node_proto, graph_proto, graph)

        return False

    @property
    def accept_versions(self) -> set:
        return set([12])
