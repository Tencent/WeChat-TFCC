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
from ir.common import create_constant_symbol
from .pattern import Pattern


class Pattern12(Pattern):
    def check_bias(self, symbol: ir.framework.Symbol):
        if len(symbol.shape) == 1:
            return True
        if symbol.is_constant():
            sub_data = symbol.data
            while len(sub_data.shape) > 1:
                sub_data = sub_data[0]
            if (sub_data == symbol.data).all():
                return True
        return False

    def create_bias_symbol(
        self, graph: ir.framework.Graph, origin_symbol: ir.framework.Symbol
    ):
        bias_name = graph.context.create_symbol_name(origin_symbol.name)

        data = origin_symbol.data
        while len(data.shape) > 1:
            data = data[0]
        create_constant_symbol(
            graph,
            bias_name,
            origin_symbol.dtype,
            ir.framework.SymbolType.CONSTANT_TENSOR,
            [origin_symbol.shape[-1]],
            data,
        )

        return bias_name

    def process_pattern(
        self, node_manager: NodeManager, rule_name: str, node_map: dict
    ):
        assert len(node_map) == 2

        output_name = node_map["add"].output_names[0]
        node_name_prefix = node_map["add"].name + "_pattern12"
        output_name_prefix = node_map["add"].output_names[0] + "_pattern12_"
        insert_index = node_manager.graph.nodes.index(node_map["add"])

        if (
            not self.check_bias(node_map["add"].inputs[1])
            or node_map["add"].inputs[1].shape[-1]
            != node_map["matmul"].inputs[1].shape[-1]
        ):
            return False

        if not node_map["add"].inputs[1].is_tensor():
            return False

        # unref
        node_manager.unrefer_node_output(node_map["add"], 0)

        bias_name = node_map["add"].input_names[1]
        if len(node_map["add"].inputs[1].shape) != 1:
            bias_name = self.create_bias_symbol(
                node_manager.graph, node_map["add"].inputs[1]
            )

        matmul_with_bias = ir.node.math.MatmulWithBias(
            node_name_prefix + "_matmul_with_bias",
            node_manager.graph,
            node_map["matmul"].input_names + [bias_name],
            [output_name],
        )
        node_manager.add_node(insert_index, matmul_with_bias)
        insert_index += 1

        return True

    @property
    def pattern_group(self):
        return PatternGroup.GROUP_1
