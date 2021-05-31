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

import ir.framework
import ir.node
from ir.optimizer.patternmanager import NodeManager
from .pattern import Pattern


class Pattern5(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2
        output_name = node_map["cast_y"].output_names[0]
        node_name = node_map["cast_y"].name + "_pattern5_cast"
        insert_index = node_manager.graph.nodes.index(node_map["cast_y"])
        dtype = node_map["cast_y"].dtype

        if node_map["cast_x"].dtype == ir.framework.DataType.BOOL:
            return False

        # unref

        node_manager.unrefer_node_output(node_map["cast_y"], 0)
        cast_node = ir.node.base.Cast(
            node_name,
            node_manager.graph,
            node_map["cast_x"].input_names,
            [output_name],
            dtype=dtype,
        )

        node_manager.add_node(insert_index, cast_node)

        return True
