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


class Pattern17(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 1
        output_name = node_map["equal_or_unequal"].output_names[0]
        node_name = node_map["equal_or_unequal"].name + "_pattern16_identity_or_not"
        insert_index = node_manager.graph.nodes.index(node_map["equal_or_unequal"])

        if node_map["equal_or_unequal"].inputs[1].shape != [1]:
            return False
        if node_map["equal_or_unequal"].inputs[1].dtype != ir.framework.DataType.BOOL:
            return False
        if not node_map["equal_or_unequal"].inputs[1].is_constant():
            return False

        target_node_type = None
        if isinstance(node_map["equal_or_unequal"], ir.node.relation.Equal):
            if node_map["equal_or_unequal"].inputs[1].data.tolist()[0]:
                target_node_type = ir.node.base.Identity
            else:
                target_node_type = ir.node.relation.Not
        elif isinstance(node_map["equal_or_unequal"], ir.node.relation.UnEqual):
            if not node_map["equal_or_unequal"].inputs[1].data.tolist()[0]:
                target_node_type = ir.node.base.Identity
            else:
                target_node_type = ir.node.relation.Not
        else:
            raise RuntimeError("Unknow error")

        # unref

        node_manager.unrefer_node_output(node_map["equal_or_unequal"], 0)
        not_equal_node = target_node_type(
            node_name,
            node_manager.graph,
            [node_map["equal_or_unequal"].input_names[0]],
            [output_name],
        )

        node_manager.add_node(insert_index, not_equal_node)

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_2
