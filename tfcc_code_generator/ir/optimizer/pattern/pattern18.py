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


class Pattern18(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 5

        output_name = node_map["to_vector"].output_names[0]
        node_name_prefix = node_map["to_vector"].name + "_pattern18"
        insert_index = node_manager.graph.nodes.index(node_map["to_vector"])
        dtype = node_map["to_vector"].outputs[0].dtype

        if len(node_map["concat"].outputs[0].shape) != 1:
            return False
        if len(node_map["concat"].input_names) != 2:
            return False
        if not node_map["concat"].inputs[1].is_constant():
            return False
        if not node_map["gather"].inputs[1].is_constant():
            return False

        # unref
        node_manager.unrefer_node_output(node_map["to_vector"], 0)

        inputs = []
        vec = []
        for idx in node_map["gather"].inputs[1].data.tolist():
            at1_node_output_name = node_manager.graph.context.create_symbol_name(
                output_name + "_at"
            )
            at1_node = ir.node.base.At1(
                node_name_prefix + "_at1_idx:" + str(idx),
                node_manager.graph,
                node_map["to_tensor"].input_names,
                [at1_node_output_name],
                idx=int(idx),
            )
            node_manager.add_node(insert_index, at1_node)
            insert_index += 1
            vec.append(len(inputs))
            inputs.append(at1_node_output_name)

        vec.append([v for v in node_map["concat"].inputs[1].data.tolist()])

        to_vector = ir.node.base.CreateVector(
            node_name_prefix + "_create_vector",
            node_manager.graph,
            inputs,
            [output_name],
            vec=vec,
            dtype=dtype,
        )
        node_manager.add_node(insert_index, to_vector)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
