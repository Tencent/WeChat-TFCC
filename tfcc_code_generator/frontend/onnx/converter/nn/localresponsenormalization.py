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


class LocalResponseNormalization(Converter):
    @property
    def op_type(self):
        return "LRN"

    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        alpha = 0.0001
        if "alpha" in attributes:
            alpha = attributes["alpha"]
        beta = 0.75
        if "beta" in attributes:
            beta = attributes["beta"]
        bias = 1.0
        if "bias" in attributes:
            bias = attributes["bias"]
        if "size" not in attributes:
            return False
        size = attributes["size"]

        lrn_node = ir.node.nn.LocalResponseNormalization(
            node_proto.name,
            graph,
            node_proto.input,
            node_proto.output,
            axis=1,
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=size,
        )
        graph.append_node(lrn_node)

        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 13])
