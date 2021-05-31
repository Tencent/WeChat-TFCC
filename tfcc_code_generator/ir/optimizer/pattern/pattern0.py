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


class Pattern0(Pattern):
    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 4
        output_name = node_map["at"].output_names[0]
        at_node = None
        node_name_prefix = node_map["at"].name + "_pattern0"
        output_name_prefix = node_map["at"].output_names[0] + "_pattern0_"
        insert_index = node_manager.graph.nodes.index(node_map["at"])

        # unref
        node_manager.unrefer_node_output(node_map["at"], 0)

        at_output_name = node_manager.graph.context.create_symbol_name(
            output_name_prefix
        )
        if isinstance(node_map["at"], ir.node.base.At):
            input_names = [
                node_map["to_tensor"].input_names[0],
                node_map["at"].input_names[1],
            ]
            at_node = ir.node.base.At(
                node_name_prefix + "_at",
                node_manager.graph,
                input_names,
                [at_output_name],
            )
        elif isinstance(node_map["at"], ir.node.base.At1):
            input_names = [
                node_map["to_tensor"].input_names[0],
            ]
            at_node = ir.node.base.At1(
                node_name_prefix + "_at",
                node_manager.graph,
                input_names,
                [at_output_name],
                idx=node_map["at"].idx,
            )
        else:
            raise RuntimeError("Internal error")

        node_manager.add_node(insert_index, at_node)
        cast_node = ir.node.base.Cast(
            node_name_prefix + "_cast",
            node_manager.graph,
            [at_output_name],
            [output_name],
            dtype=node_map["cast"].dtype,
        )
        node_manager.add_node(insert_index + 1, cast_node)

        return True
