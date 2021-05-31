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


class Unsqueeze(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        if "axes" not in attributes:
            return False
        axes = sorted(set(attributes["axes"]))
        input_symbol = graph.get_symbol(node_proto.input[0])
        target_shape_len = len(input_symbol.shape) + len(axes)
        for i, axis in enumerate(axes):
            while axis < 0:
                axis += target_shape_len
            axes[i] = axis

        if input_symbol.is_value():
            assert axes == list(range(len(axes)))
            if len(axes) == 1:
                graph.append_node(
                    ir.node.base.ToTensor(
                        node_proto.name, graph, node_proto.input, node_proto.output
                    )
                )
            else:
                to_tensor_output_name = graph.context.create_symbol_name(
                    node_proto.output[0]
                )
                graph.append_node(
                    ir.node.base.ToTensor(
                        node_proto.name + ":0",
                        graph,
                        node_proto.input,
                        [to_tensor_output_name],
                    )
                )
                graph.append_node(
                    ir.node.base.Unsqueeze(
                        node_proto.name + ":1",
                        graph,
                        [to_tensor_output_name],
                        node_proto.output,
                        axes=range(len(axes) - 1),
                    )
                )
            return True
        else:
            graph.append_node(
                ir.node.base.Unsqueeze(
                    node_proto.name,
                    graph,
                    node_proto.input,
                    node_proto.output,
                    axes=axes,
                )
            )
            return True

    @property
    def accept_versions(self) -> set:
        return set([1, 11])
