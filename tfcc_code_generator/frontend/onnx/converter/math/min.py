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


class Min(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        input_len = len(node_proto.input)
        if input_len > 1:
            node_count = input_len - 1

            maxInputName = node_proto.input[0]
            maxOutputName = graph.context.create_symbol_name(node_proto.output[0])
            for i in range(node_count):
                node_name = node_proto.name + ":" + str(i)
                if i + 1 == node_count:
                    graph.append_node(
                        ir.node.math.Min(
                            node_name,
                            graph,
                            [maxInputName, node_proto.input[i + 1]],
                            node_proto.output,
                        )
                    )
                else:
                    graph.append_node(
                        ir.node.math.Min(
                            node_name,
                            graph,
                            [maxInputName, node_proto.input[i + 1]],
                            maxOutputName,
                        )
                    )
                    maxInputName = maxOutputName
        else:
            graph.append_node(
                ir.node.base.Identity(
                    node_proto.name, graph, node_proto.input, node_proto.output
                )
            )

        return True

    @property
    def accept_versions(self):
        return set([1, 6, 8, 12])
