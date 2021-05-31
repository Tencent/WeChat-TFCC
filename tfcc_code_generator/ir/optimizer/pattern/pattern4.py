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
from .pattern import Pattern


class Pattern4(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2 or len(node_map) == 3

        if rule_name == "SRC0":
            concat_node = node_map["s0_concat"]
            to_vector_node = node_map["s0_to_vector"]
        else:
            concat_node = node_map["s1_concat"]
            to_vector_node = node_map["s1_to_vector"]

        inputs = []
        vec = []
        dtype = to_vector_node.outputs[0].dtype

        for symbol in concat_node.inputs:
            if symbol.is_constant():
                vec.append(symbol.data.tolist())
                continue
            to_tensor_node = node_manager.name_map[symbol.name]
            if isinstance(to_tensor_node, ir.node.base.Cast):
                to_tensor_node = node_manager.name_map[to_tensor_node.input_names[0]]
            if not isinstance(to_tensor_node, ir.node.base.ToTensor):
                return False
            vec.append(len(inputs))
            inputs.append(to_tensor_node.input_names[0])

        node_name = to_vector_node.name + "_pattern4_create_vector"
        output_name = to_vector_node.output_names[0]
        insert_index = node_manager.graph.nodes.index(to_vector_node)

        # unref
        node_manager.unrefer_node_output(to_vector_node, 0)

        create_vector_node = ir.node.base.CreateVector(
            node_name, node_manager.graph, inputs, [output_name], vec=vec, dtype=dtype
        )

        node_manager.add_node(insert_index, create_vector_node)

        return True
