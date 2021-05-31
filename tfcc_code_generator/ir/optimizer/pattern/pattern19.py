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


class Pattern19(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2

        squeeze_node = node_map[rule_name + "_squeeze"]
        unsqueeze_node = node_map[rule_name + "_unsqueeze"]
        if squeeze_node.axes != unsqueeze_node.axes:
            return False

        if rule_name == "SRC0":
            first_node = squeeze_node
            second_node = unsqueeze_node
        elif rule_name == "SRC1":
            first_node = unsqueeze_node
            second_node = squeeze_node
        else:
            raise RuntimeError("Unknow error")

        output_name = second_node.output_names[0]
        node_name = second_node.name + "_pattern19_identity"
        insert_index = node_manager.graph.nodes.index(first_node)

        # unref
        node_manager.unrefer_node_output(second_node, 0)

        identity = ir.node.base.Identity(
            node_name, node_manager.graph, first_node.input_names, [output_name]
        )
        node_manager.add_node(insert_index, identity)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
