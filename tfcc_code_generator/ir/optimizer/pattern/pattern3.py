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
import ir.framework
import ir.node
from ir.optimizer.patternmanager import NodeManager
from .pattern import Pattern


class Pattern3(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2

        if not isinstance(
            node_map["get_shape"].inputs[0].shape[node_map["at"].idx], int
        ):
            return False

        output_name = node_map["at"].output_names[0]

        # unref
        node_manager.unrefer_node_output(node_map["at"], 0)

        value = node_map["get_shape"].inputs[0].shape[node_map["at"].idx]
        symbol = ir.framework.Symbol(output_name)
        symbol.stype = ir.framework.SymbolType.CONSTANT_VALUE
        symbol.origin_stype = symbol.stype
        symbol.shape = [1]
        symbol.dtype = ir.framework.DataType.UINT32
        symbol.data = np.asarray([value], dtype=np.uint32)
        assert symbol.verify()
        node_manager.graph.add_symbol(symbol)
        node_manager.graph.add_keep(symbol.name)

        return True
