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
from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


class Range(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        start_to_value_name = graph.context.create_symbol_name(inp_strs[0])
        limit_to_value_name = graph.context.create_symbol_name(inp_strs[1])
        delta_to_value_name = graph.context.create_symbol_name(inp_strs[2])

        graph.append_node(
            ir.node.base.ToValue(
                op.name + ":0", graph, [inp_strs[0]], [start_to_value_name]
            )
        )
        graph.append_node(
            ir.node.base.ToValue(
                op.name + ":1", graph, [inp_strs[1]], [limit_to_value_name]
            )
        )
        graph.append_node(
            ir.node.base.ToValue(
                op.name + ":2", graph, [inp_strs[2]], [delta_to_value_name]
            )
        )

        if all(
            [
                graph.get_symbol(name).is_constant()
                for name in [
                    start_to_value_name,
                    limit_to_value_name,
                    delta_to_value_name,
                ]
            ]
        ):
            start = int(graph.get_symbol(start_to_value_name).data[0])
            limit = int(graph.get_symbol(limit_to_value_name).data[0])
            delta = int(graph.get_symbol(delta_to_value_name).data[0])

            if np.arange(start=start, stop=limit, step=delta).size == 0:
                frontend = graph.context.frontend
                if "empty_symbol" not in frontend:
                    frontend["empty_symbol"] = set()
                frontend["empty_symbol"].add(oup_strs[0])
                graph.context.frontend = frontend
                return True

        output_name = graph.context.create_symbol_name(oup_strs[0])
        graph.append_node(
            ir.node.base.Range(
                op.name + ":3",
                graph,
                [start_to_value_name, limit_to_value_name, delta_to_value_name],
                [output_name],
            )
        )
        graph.append_node(
            ir.node.base.ToTensor(op.name + ":4", graph, [output_name], oup_strs)
        )

        return True
