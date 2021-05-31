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


class Pattern1(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2
        output_name = node_map["to_y"].output_names[0]
        node_cls = node_map["to_y"].__class__
        node_name = node_map["to_y"].name + "_pattern1_to"
        insert_index = node_manager.graph.nodes.index(node_map["to_y"])

        # unref

        node_manager.unrefer_node_output(node_map["to_y"], 0)
        to_node = node_cls(
            node_name, node_manager.graph, node_map["to_x"].input_names, [output_name]
        )

        node_manager.add_node(insert_index, to_node)

        return True
