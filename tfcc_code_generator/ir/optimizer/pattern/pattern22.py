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


class Pattern22(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2

        output_name = node_map[rule_name + "_transpose_2"].output_names[0]
        node_name = node_map[rule_name + "_transpose_2"].name + "_pattern22_identity"
        insert_index = node_manager.graph.nodes.index(
            node_map[rule_name + "_transpose_2"]
        )

        if len(node_map[rule_name + "_transpose_1"].perm) != 2:
            return False

        if (
            node_map[rule_name + "_transpose_1"].perm
            != node_map[rule_name + "_transpose_2"].perm
        ):
            return False

        # unref
        node_manager.unrefer_node_output(node_map[rule_name + "_transpose_2"], 0)

        rsqrt = ir.node.base.Identity(
            node_name,
            node_manager.graph,
            node_map[rule_name + "_transpose_1"].input_names,
            [output_name],
        )
        node_manager.add_node(insert_index, rsqrt)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
