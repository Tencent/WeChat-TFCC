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


class Pattern15(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 11 or (rule_name == "SRC3" and len(node_map) == 9)

        rule_name = rule_name.split(":")[0]
        output_name = node_map[rule_name + "_result"].output_names[0]
        node_name = (
            node_map[rule_name + "_result"].name + "_pattern15_layer_normalization"
        )
        insert_index = node_manager.graph.nodes.index(node_map[rule_name + "_result"])

        axes = node_map[rule_name + "_mean"].axes
        input_symbol = node_map[rule_name + "_mean"].inputs[0]
        if len(axes) != 1:
            return False
        if axes != list(
            range(len(input_symbol.shape) - len(axes), len(input_symbol.shape))
        ):
            return False
        if axes != node_map[rule_name + "_variance"].axes:
            return False
        variance_symbol = node_map[rule_name + "_variance"].outputs[0]
        if node_map[rule_name + "_add_0"].input_names[0] == variance_symbol.name:
            epsilon_symbol = node_map[rule_name + "_add_0"].inputs[1]
        else:
            epsilon_symbol = node_map[rule_name + "_add_0"].inputs[0]
        if epsilon_symbol.shape != [1] or not epsilon_symbol.is_constant():
            return False
        if rule_name == "SRC3":
            if node_map["SRC3_x"].input_names[0] == node_map["SRC3_mx"].output_names[0]:
                gamma_symbol = node_map["SRC3_x"].inputs[1]
            else:
                gamma_symbol = node_map["SRC3_x"].inputs[0]
        else:
            inv_symbol = node_map[rule_name + "_inv"].outputs[0]
            if node_map[rule_name + "_x"].input_names[0] == inv_symbol.name:
                gamma_symbol = node_map[rule_name + "_x"].inputs[1]
            else:
                gamma_symbol = node_map[rule_name + "_x"].inputs[0]

        if rule_name == "SRC0":
            if (
                node_map["SRC0_add_1"].input_names[0]
                == node_map["SRC0_ix"].output_names[0]
            ):
                beta_symbol = node_map["SRC0_add_1"].inputs[1]
            else:
                beta_symbol = node_map["SRC0_add_1"].inputs[0]
        if rule_name == "SRC1":
            if (
                node_map["SRC1_result"].input_names[0]
                == node_map["SRC1_sub_1"].output_names[0]
            ):
                beta_symbol = node_map["SRC1_result"].inputs[1]
            else:
                beta_symbol = node_map["SRC1_result"].inputs[0]
        if rule_name == "SRC2":
            beta_symbol = node_map["SRC2_sub_1"].inputs[0]
        if rule_name == "SRC3":
            if (
                node_map["SRC3_result"].input_names[0]
                == node_map["SRC3_x"].output_names[0]
            ):
                beta_symbol = node_map["SRC3_result"].inputs[1]
            else:
                beta_symbol = node_map["SRC3_result"].inputs[0]

        if beta_symbol.shape != gamma_symbol.shape:
            return False
        if beta_symbol.shape != input_symbol.shape[axes[0] :]:
            return False

        # unref
        node_manager.unrefer_node_output(node_map[rule_name + "_result"], 0)

        inputs = [
            input_symbol.name,
            gamma_symbol.name,
            beta_symbol.name,
        ]
        layer_normalization_node = ir.node.nn.LayerNormalization(
            node_name,
            node_manager.graph,
            inputs,
            [output_name],
            epsilon=epsilon_symbol.data.tolist()[0],
        )

        node_manager.add_node(insert_index, layer_normalization_node)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_2
