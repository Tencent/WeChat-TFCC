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


class Dropout(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        if len(node_proto.output) > 2:
            return False
        if len(node_proto.output) == 2:
            mask_name = node_proto.output[1]
            if mask_name in [proto.name for proto in graph_proto.output]:
                return False
            input_name_set = set()
            for proto in graph_proto.node:
                input_name_set.update(proto.input)
            if mask_name in input_name_set:
                return False

        graph.append_node(
            ir.node.base.Identity(
                node_proto.name, graph, [node_proto.input[0]], [node_proto.output[0]]
            )
        )
        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 6, 7, 10, 12])
