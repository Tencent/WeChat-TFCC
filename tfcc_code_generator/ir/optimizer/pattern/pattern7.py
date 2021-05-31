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


class Pattern7(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 3 or len(node_map) == 4

        to_tensor_node = node_map[rule_name + "_to_tensor"]
        slice_or_gather_node = node_map[rule_name + "_slicev2_or_gather"]
        to_value_node = node_map[rule_name + "_to_value"]

        output_name = to_value_node.output_names[0]
        dtype = to_value_node.outputs[0].dtype
        input_dtype = to_tensor_node.inputs[0].dtype
        node_name_prefix = to_value_node.name + "_pattern7"
        output_name_prefix = to_value_node.output_names[0] + "_pattern7_"
        insert_index = node_manager.graph.nodes.index(to_value_node)

        if not slice_or_gather_node.inputs[1].is_constant():
            return False

        idx_value = slice_or_gather_node.inputs[1].data.tolist()[0]
        if idx_value < 0:
            if not isinstance(to_tensor_node.inputs[0].shape[0], int):
                return False
            while idx_value < 0:
                idx_value += to_tensor_node.inputs[0].shape[0]

        # unref
        node_manager.unrefer_node_output(to_value_node, 0)

        if dtype == input_dtype:
            at_node_output_name = output_name
        else:
            at_node_output_name = node_manager.graph.context.create_symbol_name(
                output_name_prefix
            )
        at_node_input_names = [to_tensor_node.input_names[0]]
        at_node = ir.node.base.At1(
            node_name_prefix + "_at1",
            node_manager.graph,
            at_node_input_names,
            [at_node_output_name],
            idx=idx_value,
        )
        node_manager.add_node(insert_index, at_node)
        insert_index += 1

        if dtype != input_dtype:
            cast_node = ir.node.base.Cast(
                node_name_prefix + "_cast",
                node_manager.graph,
                [at_node_output_name],
                [output_name],
                dtype=dtype,
            )
            node_manager.add_node(insert_index, cast_node)
            insert_index += 1

        return True
