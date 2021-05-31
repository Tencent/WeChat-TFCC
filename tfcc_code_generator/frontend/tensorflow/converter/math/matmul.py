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

from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/MatMul?hl=en
class MatMul(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        assert len(inp_strs) == 2 and len(oup_strs) == 1
        a_name = inp_strs[0]
        b_name = inp_strs[1]
        transpose_a = op.get_attr("transpose_a")
        transpose_b = op.get_attr("transpose_b")

        if transpose_a:
            transpose_output_name = graph.context.create_symbol_name(a_name)
            perm = list(range(len(graph.get_symbol(a_name).shape) - 2))
            perm = perm + [len(perm) + 1, len(perm)]
            graph.append_node(
                ir.node.base.Transpose(
                    op.name + ":transpose_a",
                    graph,
                    [a_name],
                    [transpose_output_name],
                    perm=perm,
                )
            )
            a_name = transpose_output_name
        if transpose_b:
            transpose_output_name = graph.context.create_symbol_name(b_name)
            perm = list(range(len(graph.get_symbol(b_name).shape) - 2))
            perm = perm + [len(perm) + 1, len(perm)]
            graph.append_node(
                ir.node.base.Transpose(
                    op.name + ":transpose_a",
                    graph,
                    [b_name],
                    [transpose_output_name],
                    perm=perm,
                )
            )
            b_name = transpose_output_name

        graph.append_node(
            ir.node.math.Matmul(op.name, graph, [a_name, b_name], oup_strs)
        )
        return True
