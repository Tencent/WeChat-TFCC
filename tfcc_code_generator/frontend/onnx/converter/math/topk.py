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
from ir.common import create_constant_symbol


class TopK(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        axis = -1
        if "axis" in attributes:
            axis = attributes["axis"]

        if "largest" in attributes and attributes["largest"] != 1:
            return False

        symbol = graph.get_symbol(node_proto.input[0])
        while axis < 0:
            axis += len(symbol.shape)

        if axis != len(symbol.shape) - 1:
            return False

        if self.since_version == 1:
            if "k" not in attributes:
                return False
            k_name = graph.context.create_symbol_name(node_proto.output[0] + "_k")
            k = attributes["k"]
            create_constant_symbol(
                graph,
                k_name,
                ir.framework.DataType.UINT32,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([k], dtype=np.uint32),
            )
            graph.append_node(
                ir.node.math.TopK(
                    node_proto.name,
                    graph,
                    [node_proto.input[0], k_name],
                    node_proto.output,
                )
            )
        else:
            to_value_output_name = graph.context.create_symbol_name(node_proto.input[1])

            graph.append_node(
                ir.node.base.ToValue(
                    node_proto.name + ":0",
                    graph,
                    [node_proto.input[1]],
                    [to_value_output_name],
                )
            )
            graph.append_node(
                ir.node.math.TopK(
                    node_proto.name + ":1",
                    graph,
                    [node_proto.input[0], to_value_output_name],
                    node_proto.output,
                )
            )

        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 10, 11])
