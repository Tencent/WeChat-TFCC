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
from ir.optimizer.patterngroup import PatternGroup
from .pattern import Pattern


class Pattern16(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2
        output_name = node_map["not"].output_names[0]
        node_name = node_map["not"].name + "_pattern16_not_equal"
        insert_index = node_manager.graph.nodes.index(node_map["not"])

        # unref

        node_manager.unrefer_node_output(node_map["not"], 0)
        not_equal_node = ir.node.relation.UnEqual(
            node_name, node_manager.graph, node_map["equal"].input_names, [output_name]
        )

        node_manager.add_node(insert_index, not_equal_node)

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
