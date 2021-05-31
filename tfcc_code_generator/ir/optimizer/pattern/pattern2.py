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
from ir.optimizer.patterngroup import PatternGroup
from .pattern import Pattern


class Pattern2(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 1

        if not node_map["at"].inputs[1].is_constant():
            return False

        output_name = node_map["at"].output_names[0]
        node_name = node_map["at"].name + "_pattern2_at"
        insert_index = node_manager.graph.nodes.index(node_map["at"])

        # unref
        node_manager.unrefer_node_output(node_map["at"], 0)

        idx = node_map["at"].inputs[1].data.tolist()[0]
        to_node = ir.node.base.At1(
            node_name,
            node_manager.graph,
            [node_map["at"].input_names[0]],
            [output_name],
            idx=idx,
        )

        node_manager.add_node(insert_index, to_node)

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
