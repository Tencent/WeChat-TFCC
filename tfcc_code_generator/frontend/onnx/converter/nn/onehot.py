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


class OneHot(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        depth_symbol = graph.get_symbol(node_proto.input[1])
        if not depth_symbol.is_constant():
            return False
        values_symbol = graph.get_symbol(node_proto.input[2])
        if not values_symbol.is_constant():
            return False
        if values_symbol.data.size != 2:
            return False

        attributes = self.get_attributes(node_proto)
        if "axis" in attributes:
            if attributes["axis"] != -1:
                return False

        depth = depth_symbol.data.tolist()[0]
        off_value = values_symbol.data.tolist()[0]
        on_value = values_symbol.data.tolist()[1]

        one_hot_node = ir.node.nn.OneHot(
            node_proto.name,
            graph,
            [node_proto.input[0]],
            node_proto.output,
            dtype=values_symbol.dtype,
            depth=depth,
            off_value=off_value,
            on_value=on_value,
        )
        graph.append_node(one_hot_node)

        return True

    @property
    def accept_versions(self) -> set:
        return set([9, 11])
