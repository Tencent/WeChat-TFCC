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


class Pattern20(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 3

        slice_0_node = node_map[rule_name + "_slice_0"]
        slice_1_node = node_map[rule_name + "_slice_1"]
        concat_node = node_map[rule_name + "_concat"]

        if slice_0_node.input_names[0] != slice_1_node.input_names[0]:
            return False
        if not all([isinstance(s, int) for s in slice_0_node.inputs[0].shape]):
            return False
        if slice_0_node.length != slice_1_node.length:
            return False
        if slice_0_node.axis != slice_1_node.axis:
            return False
        if slice_0_node.length * 2 != slice_0_node.inputs[0].shape[slice_0_node.axis]:
            return False
        if (
            not slice_0_node.input[1].is_constant()
            and slice_0_node.input[1].data.tolist()[0] != 0
        ):
            return False
        if (
            not slice_1_node.input[1].is_constant()
            and slice_1_node.input[1].data.tolist()[0] != slice_0_node.length
        ):
            return False

        slice_axis = slice_0_node.axis
        concat_axis = concat_node.axis

        output_name = concat_node.output_names[0]
        node_name_prefix = concat_node.name + "_pattern20"
        insert_index = node_manager.graph.nodes.index(concat_node)

        # unref
        node_manager.unrefer_node_output(concat_node, 0)

        identity = ir.node.base.Identity(
            node_name, node_manager.graph, first_node.input_names, [output_name]
        )
        node_manager.add_node(insert_index, identity)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
