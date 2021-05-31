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


class Pattern9(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 4 or len(node_map) == 5 or len(node_map) == 6

        to_tensor_node = node_map[rule_name + "_to_tensor"]
        gather_node = node_map[rule_name + "_gather"]
        reduce_node = node_map[rule_name + "_reduce"]
        to_value_node = node_map[rule_name + "_to_value"]

        if not gather_node.inputs[1].is_constant():
            return False

        output_name = to_value_node.output_names[0]
        dtype = to_value_node.outputs[0].dtype
        node_name_prefix = to_value_node.name + "_pattern9"
        output_name_prefix = to_value_node.output_names[0] + "_pattern9_"
        insert_index = node_manager.graph.nodes.index(to_value_node)

        if isinstance(reduce_node, ir.node.math.ReduceSum):
            calculate_cls = ir.node.math.Add
        elif isinstance(reduce_node, ir.node.math.ReduceProd):
            calculate_cls = ir.node.math.Mul
        else:
            raise RuntimeError("Unknow error")

        # unref
        node_manager.unrefer_node_output(to_value_node, 0)

        names = []
        at_node_count = 0
        for idx in gather_node.inputs[1].data.tolist():
            at_node_output_name = node_manager.graph.context.create_symbol_name(
                output_name_prefix
            )
            at_node = ir.node.base.At1(
                node_name_prefix + "_at_" + str(at_node_count),
                node_manager.graph,
                to_tensor_node.input_names,
                [at_node_output_name],
                idx=idx,
            )
            at_node_count += 1
            node_manager.add_node(insert_index, at_node)
            insert_index += 1
            names.append(at_node_output_name)

        calculate_output_name = node_manager.graph.context.create_symbol_name(
            output_name_prefix
        )
        if len(names) == 1:
            identity_node = ir.node.base.Identity(
                node_name_prefix + "_identity",
                node_manager.graph,
                names,
                [calculate_output_name],
            )
            node_manager.add_node(insert_index, identity_node)
            insert_index += 1
        elif len(names) > 1:
            calculate_count = 0
            calculate_node = calculate_cls(
                node_name_prefix + "_calculate_" + str(calculate_count),
                node_manager.graph,
                names[:2],
                [calculate_output_name],
            )
            calculate_count += 1
            node_manager.add_node(insert_index, calculate_node)
            insert_index += 1
            for name in names[2:]:
                new_calculate_output_name = (
                    node_manager.graph.context.create_symbol_name(output_name_prefix)
                )
                calculate_node = calculate_cls(
                    node_name_prefix + "_calculate_" + str(calculate_count),
                    node_manager.graph,
                    [calculate_output_name, name],
                    [new_calculate_output_name],
                )
                calculate_count += 1
                node_manager.add_node(insert_index, calculate_node)
                insert_index += 1
                calculate_output_name = new_calculate_output_name
        else:
            raise RuntimeError("Unknow error")

        cast_node = ir.node.base.Cast(
            node_name_prefix + "_cast",
            node_manager.graph,
            [calculate_output_name],
            [output_name],
            dtype=dtype,
        )
        node_manager.add_node(insert_index, cast_node)
        insert_index += 1

        return True
