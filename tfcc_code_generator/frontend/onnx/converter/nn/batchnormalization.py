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


class BatchNormalization(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False
        attributes = self.get_attributes(node_proto)
        if "spatial" in attributes and attributes["spatial"] != 1:
            for i in range(1, len(node_proto.input)):
                if len(graph.get_symbol(node_proto.input[i]).shape) > 1:
                    return False
        epsilon = 1e-05
        if "epsilon" in attributes:
            epsilon = attributes["epsilon"]

        if len(node_proto.output) > 1:
            used_name_set = set([proto.name for proto in graph_proto.output])
            for proto in graph_proto.node:
                used_name_set.update(proto.input)
            for name in graph_proto.output[1:]:
                if name in used_name_set:
                    return False

        graph.append_node(
            ir.node.nn.BatchNormalization(
                node_proto.name,
                graph,
                node_proto.input,
                [node_proto.output[0]],
                axis=1,
                epsilon=epsilon,
            )
        )

        return True

    @property
    def accept_versions(self) -> set:
        return set([7, 9])
