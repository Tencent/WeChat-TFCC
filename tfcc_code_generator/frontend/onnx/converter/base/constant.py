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


class Constant(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        if "value" not in attributes:
            return False

        symbol = frontend.onnx.common.onnx_tensor_to_symbol(
            attributes["value"], node_proto.output[0]
        )

        graph.add_symbol(symbol)
        graph.add_keep(symbol.name)
        return True

    @property
    def accept_versions(self) -> set:
        # TODO support attributes sparse_value and value_*
        return set([1, 9, 11, 12])
