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

import numpy as np
import onnx
import ir.node
import ir.framework
import frontend.onnx.common
from ..converter import Converter


class MatrixBandPart(Converter):
    def __init__(self, op_set: dict):
        self._valid = True

    @property
    def domain(self):
        return "ai.onnx.contrib"

    def accept(self, node_proto: onnx.NodeProto):
        if node_proto.domain != self.domain:
            return False
        if node_proto.op_type != self.op_type:
            return False
        if len(node_proto.input) != 3 or len(node_proto.output) != 1:
            return False
        return True

    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        num_lower_symbol = graph.get_symbol(node_proto.input[1])
        num_upper_symbol = graph.get_symbol(node_proto.input[2])

        if not num_lower_symbol.is_constant() or not num_upper_symbol.is_constant():
            return False

        if num_lower_symbol.data.size > 1 or num_upper_symbol.data.size > 1:
            return False

        if (
            num_lower_symbol.data.tolist()[0] < 0
            and num_upper_symbol.data.tolist()[0] < 0
        ):
            graph.append_node(
                ir.node.base.Identity(
                    node_proto.name, graph, [node_proto.input[0]], node_proto.output
                )
            )
        elif (
            num_lower_symbol.data.tolist()[0] < 0
            and num_upper_symbol.data.tolist()[0] >= 0
        ):
            k = num_upper_symbol.data.tolist()[0]
            graph.append_node(
                ir.node.base.Tril(
                    node_proto.name,
                    graph,
                    [node_proto.input[0]],
                    node_proto.output,
                    k=k,
                )
            )
        elif (
            num_lower_symbol.data.tolist()[0] >= 0
            and num_upper_symbol.data.tolist()[0] < 0
        ):
            k = -num_lower_symbol.data.tolist()[0]
            graph.append_node(
                ir.node.base.Triu(
                    node_proto.name,
                    graph,
                    [node_proto.input[0]],
                    node_proto.output,
                    k=k,
                )
            )
        else:
            k_tril = num_upper_symbol.data.tolist()[0]
            k_triu = -num_lower_symbol.data.tolist()[0]

            tmp_output_name = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                ir.node.base.Tril(
                    node_proto.name,
                    graph,
                    [node_proto.input[0]],
                    [tmp_output_name],
                    k=k_tril,
                )
            )
            graph.append_node(
                ir.node.base.Triu(
                    node_proto.name,
                    graph,
                    [tmp_output_name],
                    node_proto.output,
                    k=k_triu,
                )
            )
        return True
