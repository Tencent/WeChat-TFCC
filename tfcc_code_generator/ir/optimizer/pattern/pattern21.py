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


class Pattern21(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 5
        rule_name = rule_name.split(":")[0]

        output_name = node_map[rule_name + "_mul2"].output_names[0]
        node_name = node_map[rule_name + "_mul2"].name + "_pattern21_gelu"
        insert_index = node_manager.graph.nodes.index(node_map[rule_name + "_mul2"])

        input_symbol = node_map[rule_name + "_div"].inputs[0]
        dtype = input_symbol.dtype.numpy_dtype
        if not node_map[rule_name + "_div"].inputs[1].is_constant():
            return False
        if not node_map[rule_name + "_div"].inputs[1].is_value():
            return False
        if (
            node_map[rule_name + "_div"].inputs[1].data.tolist()[0]
            != np.asarray([np.sqrt(2)], dtype=dtype).tolist()[0]
        ):
            return False

        constant_one_symbol = node_map[rule_name + "_add"].inputs[1]
        if (
            node_map[rule_name + "_add"].input_names[1]
            == node_map[rule_name + "_erf"].output_names[0]
        ):
            constant_one_symbol = node_map[rule_name + "_add"].inputs[0]

        if not constant_one_symbol.is_constant() or not constant_one_symbol.is_value():
            return False
        if constant_one_symbol.data.tolist()[0] != 1:
            return False

        if (
            node_map[rule_name + "_add"].output_names[0]
            == node_map[rule_name + "_mul"].input_names[0]
        ):
            if node_map[rule_name + "_mul"].input_names[1] != input_symbol.name:
                return False
        else:
            if node_map[rule_name + "_mul"].input_names[0] != input_symbol.name:
                return False

        constant_p5_symbol = node_map[rule_name + "_mul2"].inputs[1]
        if (
            node_map[rule_name + "_mul2"].input_names[1]
            == node_map[rule_name + "_mul"].output_names[0]
        ):
            constant_p5_symbol = node_map[rule_name + "_mul2"].inputs[0]
        if (
            constant_p5_symbol.data.tolist()[0]
            != np.asarray([0.5], dtype=dtype).tolist()[0]
        ):
            return False

        # unref

        node_manager.unrefer_node_output(node_map[rule_name + "_mul2"], 0)
        not_equal_node = ir.node.math.Gelu(
            node_name, node_manager.graph, [input_symbol.name], [output_name]
        )

        node_manager.add_node(insert_index, not_equal_node)

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
