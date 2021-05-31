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


class Shape(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        get_shape_output = graph.context.create_symbol_name(node_proto.output[0])
        graph.append_node(
            ir.node.base.GetShape(
                node_proto.name + ":0", graph, node_proto.input, [get_shape_output]
            )
        )
        to_tensor_output = graph.context.create_symbol_name(node_proto.output[0])
        graph.append_node(
            ir.node.base.ToTensor(
                node_proto.name + ":1", graph, [get_shape_output], [to_tensor_output]
            )
        )
        graph.append_node(
            ir.node.base.Cast(
                node_proto.name + ":2",
                graph,
                [to_tensor_output],
                node_proto.output,
                dtype=ir.framework.DataType.INT64,
            )
        )
        return True

    @property
    def accept_versions(self) -> set:
        return set([1])
