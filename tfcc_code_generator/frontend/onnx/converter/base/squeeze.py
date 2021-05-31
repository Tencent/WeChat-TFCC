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


class Squeeze(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        input_symbol = graph.get_symbol(node_proto.input[0])
        if "axes" not in attributes and 1 not in input_symbol.shape:
            return False
        if "axes" in attributes:
            axes = list(set(attributes["axes"]))
        else:
            axes = [i for i, s in enumerate(input_symbol.shape) if s == 1]
        for i in range(0, len(axes)):
            if axes[i] < 0:
                axes[i] = axes[i] + len(input_symbol.shape)
        axes = sorted(axes)

        if axes == list(range(len(input_symbol.shape))):
            graph.append_node(
                ir.node.base.ToValue(
                    node_proto.name, graph, node_proto.input, node_proto.output
                )
            )
            return True
        else:
            graph.append_node(
                ir.node.base.Squeeze(
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
