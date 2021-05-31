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


class Reduce(Converter):
    def __init__(self, op_set, node_cls):
        super().__init__(op_set)
        self._node_cls = node_cls

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
        keep_dims = True
        if "keepdims" in attributes:
            keep_dims = attributes["keepdims"] > 0

        axes = []
        symbol = graph.get_symbol(node_proto.input[0])
        for axis in attributes["axes"]:
            while axis < 0:
                axis += len(symbol.shape)
            axes.append(axis)

        if keep_dims:
            graph.append_node(
                self._node_cls(
                    node_proto.name,
                    graph,
                    node_proto.input,
                    node_proto.output,
                    axes=axes,
                )
            )
        else:
            squeeze_output_name = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                self._node_cls(
                    node_proto.name + ":0",
                    graph,
                    node_proto.input,
                    [squeeze_output_name],
                    axes=axes,
                )
            )
            if list(range(len(symbol.shape))) == list(sorted(axes)):
                graph.append_node(
                    ir.node.base.ToValue(
                        node_proto.name + ":1",
                        graph,
                        [squeeze_output_name],
                        node_proto.output,
                    )
                )
            else:
                graph.append_node(
                    ir.node.base.Squeeze(
                        node_proto.name + ":1",
                        graph,
                        [squeeze_output_name],
                        node_proto.output,
                        axes=axes,
                    )
                )

        return True


class ReduceMean(Reduce):
    def __init__(self, op_set: dict):
        super().__init__(op_set, ir.node.math.ReduceMean)

    @property
    def accept_versions(self) -> set:
        return set([1, 11])


class ReduceSum(Reduce):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.math.ReduceSum)

    @property
    def accept_versions(self) -> set:
        return set([1, 11])


class ReduceProd(Reduce):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.math.ReduceProd)

    @property
    def accept_versions(self) -> set:
        return set([1, 11])


class ReduceMax(Reduce):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.math.ReduceMax)

    @property
    def accept_versions(self) -> set:
        return set([1, 11])


class ReduceMin(Reduce):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.math.ReduceMin)

    @property
    def accept_versions(self) -> set:
        return set([1, 11])
