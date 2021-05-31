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


class Relation(Converter):
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

        graph.append_node(
            self._node_cls(node_proto.name, graph, node_proto.input, node_proto.output)
        )
        return True


class Equal(Relation):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.relation.Equal)

    @property
    def accept_versions(self) -> set:
        return set([7, 11])


class Less(Relation):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.relation.Less)

    @property
    def accept_versions(self) -> set:
        return set([7, 9])


class LessOrEqual(Relation):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.relation.LessOrEqual)

    @property
    def accept_versions(self) -> set:
        return set([12])


class Greater(Relation):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.relation.Greater)

    @property
    def accept_versions(self) -> set:
        return set([7, 9])


class GreaterOrEqual(Relation):
    def __init__(self, op_set):
        super().__init__(op_set, ir.node.relation.GreaterOrEqual)

    @property
    def accept_versions(self) -> set:
        return set([12])
