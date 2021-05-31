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


class Clip(Converter):
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
        if input_symbol.is_integer():
            lowest_value = np.iinfo(input_symbol.dtype.numpy_dtype).min
            max_value = np.iinfo(input_symbol.dtype.numpy_dtype).max
        else:
            lowest_value = np.finfo(input_symbol.dtype.numpy_dtype).min
            max_value = np.finfo(input_symbol.dtype.numpy_dtype).max

        if "min" in attributes:
            lowest_value = attributes["min"]
        if "max" in attributes:
            max_value = attributes["max"]

        if len(node_proto.input) >= 2:
            min_to_value_name = graph.context.create_symbol_name(node_proto.input[1])
            graph.append_node(
                ir.node.base.ToValue(
                    node_proto.name + ":0",
                    graph,
                    [node_proto.input[1]],
                    [min_to_value_name],
                )
            )
        else:
            min_to_value_name = graph.context.create_symbol_name(node_proto.output[0])
            create_constant_symbol(
                graph,
                min_to_value_name,
                input_symbol.dtype,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([lowest_value], dtype=input_symbol.dtype.numpy_dtype),
            )

        if len(node_proto.input) >= 3:
            max_to_value_name = graph.context.create_symbol_name(node_proto.input[2])
            graph.append_node(
                ir.node.base.ToValue(
                    node_proto.name + ":1",
                    graph,
                    [node_proto.input[2]],
                    [max_to_value_name],
                )
            )
        else:
            max_to_value_name = graph.context.create_symbol_name(node_proto.output[0])
            create_constant_symbol(
                graph,
                max_to_value_name,
                input_symbol.dtype,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([max_value], dtype=input_symbol.dtype.numpy_dtype),
            )

        graph.append_node(
            ir.node.math.Clip(
                node_proto.name + ":2",
                graph,
                [node_proto.input[0], min_to_value_name, max_to_value_name],
                node_proto.output,
            )
        )

        return True

    @property
    def accept_versions(self) -> set:
        # TODO support Clip-6
        return set([6, 11, 12])
