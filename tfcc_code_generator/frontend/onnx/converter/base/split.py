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


class Split(Converter):
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
        while axis < 0:
            axis += len(input_symbol.shape)
        if "split" in attributes and set(attributes["split"]) != set([1]):
            start = 0
            for i, length in enumerate(attributes["split"]):
                start_name = graph.context.create_symbol_name(
                    node_proto.output[0] + "_start_"
                )
                create_constant_symbol(
                    graph,
                    start_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([start], dtype=np.uint32),
                )
                graph.append_node(
                    ir.node.base.SliceV1(
                        node_proto.name + ":" + str(i),
                        graph,
                        [node_proto.input[0], start_name],
                        [node_proto.output[i]],
                        axis=axis,
                        length=length,
                    )
                )
                start += length
        else:
            graph.append_node(
                ir.node.base.Split(
                    node_proto.name,
                    graph,
                    node_proto.input,
                    node_proto.output,
                    axis=axis,
                )
            )
        return True

    @property
    def accept_versions(self) -> set:
        # TODO support Split-1
        return set([2, 11])
