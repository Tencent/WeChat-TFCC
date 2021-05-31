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


class Gather(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        axis = 0
        if "axis" in attributes:
            axis = attributes["axis"]

        input_symbol = graph.get_symbol(node_proto.input[0])
        indices_symbol = graph.get_symbol(node_proto.input[1])
        while axis < 0:
            axis = axis + len(input_symbol.shape)

        if input_symbol.is_tensor() and indices_symbol.is_tensor():
            graph.append_node(
                ir.node.nn.Gather(
                    node_proto.name,
                    graph,
                    node_proto.input,
                    node_proto.output,
                    axis=axis,
                )
            )
            return True
        elif (
            input_symbol.is_tensor()
            and indices_symbol.is_value()
            and len(input_symbol.shape) == 1
        ):
            toVectorOutputName = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                ir.node.base.ToVector(
                    node_proto.name + ":0",
                    graph,
                    [node_proto.input[0]],
                    [toVectorOutputName],
                )
            )
            graph.append_node(
                ir.node.base.At(
                    node_proto.name + ":1",
                    graph,
                    [toVectorOutputName, node_proto.input[1]],
                    node_proto.output,
                )
            )
            return True
        elif (
            input_symbol.is_tensor()
            and indices_symbol.is_value()
            and len(input_symbol.shape) > 1
        ):
            sliceOutputName = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                ir.node.base.SliceV1(
                    node_proto.name + ":0",
                    graph,
                    node_proto.input,
                    [sliceOutputName],
                    axis=axis,
                    length=1,
                )
            )
            graph.append_node(
                ir.node.base.Squeeze(
                    node_proto.name + ":1",
                    graph,
                    [sliceOutputName],
                    node_proto.output,
                    axes=[axis],
                )
            )
            return True
        else:
            return False

    @property
    def accept_versions(self) -> set:
        # TODO Support negative indices
        return set([1, 11])
