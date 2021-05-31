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


class Pattern10(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 1

        if not node_map["pow"].inputs[1].is_constant():
            return False
        exponent = node_map["pow"].inputs[1].data.tolist()[0]
        if int(exponent) != exponent:
            return False

        exponent = int(exponent)
        if exponent > 4:
            return False

        exponent_map = {1: node_map["pow"].input_names[0]}

        output_name = node_map["pow"].output_names[0]
        node_name_prefix = node_map["pow"].name + "_pattern10_"
        node_index = 0
        insert_index = node_manager.graph.nodes.index(node_map["pow"])

        # unref
        node_manager.unrefer_node_output(node_map["pow"], 0)

        sub_output_name, _, _, insert_index = self.process_mul(
            node_manager,
            exponent,
            output_name + "_tmp",
            exponent_map,
            node_name_prefix,
            node_index,
            insert_index,
        )
        node_manager.add_node(
            insert_index,
            ir.node.base.Identity(
                node_name_prefix + "identity",
                node_manager.graph,
                [sub_output_name],
                [output_name],
            ),
        )
        insert_index += 1

        return True

    def process_mul(
        self,
        node_manager: NodeManager,
        exponent: int,
        output_name: str,
        exponent_map: dict,
        node_name_prefix: str,
        node_index: int,
        insert_index: int,
    ):
        if exponent in exponent_map:
            return exponent_map[exponent], exponent_map, node_index, insert_index
        first_exponent = exponent // 2
        first_input_name, exponent_map, node_index, insert_index = self.process_mul(
            node_manager,
            first_exponent,
            output_name,
            exponent_map,
            node_name_prefix,
            node_index,
            insert_index,
        )

        second_exponent = exponent - first_exponent
        second_input_name, exponent_map, node_index, insert_index = self.process_mul(
            node_manager,
            second_exponent,
            output_name,
            exponent_map,
            node_name_prefix,
            node_index,
            insert_index,
        )

        output_name = node_manager.graph.context.create_symbol_name(output_name)
        mul_node = ir.node.math.Mul(
            node_name_prefix + "mul_" + str(node_index),
            node_manager.graph,
            [first_input_name, second_input_name],
            [output_name],
        )
        node_index += 1
        node_manager.add_node(insert_index, mul_node)
        insert_index += 1
        exponent_map[exponent] = output_name
        return output_name, exponent_map, node_index, insert_index

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
