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


class Gemm(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        a_name = node_proto.input[0]
        b_name = node_proto.input[1]

        attributes = self.get_attributes(node_proto)
        node_count = 0
        if "transA" in attributes and attributes["transA"]:
            transposeOutputName = graph.context.create_symbol_name(node_proto.output[0])
            node_name = node_proto.name
            if node_count > 0:
                node_name = node_name + ":" + str(node_count)
            node_count += 1
            symbol = graph.get_symbol(a_name)
            perm = list(range(len(symbol.shape) - 2))
            perm = perm + [len(perm) + 1, len(perm)]
            graph.append_node(
                ir.node.base.Transpose(
                    node_name, graph, [a_name], [transposeOutputName], perm=perm
                )
            )
            a_name = transposeOutputName

        if "transB" in attributes and attributes["transB"]:
            transposeOutputName = graph.context.create_symbol_name(node_proto.output[0])
            node_name = node_proto.name
            if node_count > 0:
                node_name = node_name + ":" + str(node_count)
            node_count += 1
            symbol = graph.get_symbol(b_name)
            perm = list(range(len(symbol.shape) - 2))
            perm = perm + [len(perm) + 1, len(perm)]
            graph.append_node(
                ir.node.base.Transpose(
                    node_name, graph, [b_name], [transposeOutputName], perm=perm
                )
            )
            b_name = transposeOutputName

        node_name = node_proto.name
        if node_count > 0:
            node_name = node_name + ":" + str(node_count)
        node_count += 1
        if len(node_proto.input) == 3:
            graph.append_node(
                ir.node.math.MatmulWithBias(
                    node_name,
                    graph,
                    [a_name, b_name, node_proto.input[2]],
                    node_proto.output,
                )
            )
        else:
            graph.append_node(
                ir.node.math.Matmul(
                    node_name, graph, [a_name, b_name], node_proto.output
                )
            )
        return True

    @property
    def accept_versions(self) -> set:
        return set([7, 9, 11])
