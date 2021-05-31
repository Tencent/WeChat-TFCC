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


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/BatchMatMul?hl=en
class BatchMatMul(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        adj_x = op.get_attr("adj_x")
        adj_y = op.get_attr("adj_y")

        input_symbol = graph.get_symbol(inp_strs[0])
        kernel_symbol = graph.get_symbol(inp_strs[1])

        index = 0
        input_name = inp_strs[0]
        kernel_name = inp_strs[1]
        if adj_x:
            transpose_output_name = graph.context.create_symbol_name(input_name)
            perm = list(range(len(input_symbol.shape)))
            perm[-1] = len(input_symbol.shape) - 2
            perm[-2] = len(input_symbol.shape) - 1
            graph.append_node(
                ir.node.base.Transpose(
                    op.name + ":" + str(index),
                    graph,
                    [input_name],
                    [transpose_output_name],
                    perm=perm,
                )
            )
            index += 1
            input_name = transpose_output_name
        if adj_y:
            transpose_output_name = graph.context.create_symbol_name(kernel_name)
            perm = list(range(len(kernel_symbol.shape)))
            perm[-1] = len(kernel_symbol.shape) - 2
            perm[-2] = len(kernel_symbol.shape) - 1
            graph.append_node(
                ir.node.base.Transpose(
                    op.name + ":" + str(index),
                    graph,
                    [kernel_name],
                    [transpose_output_name],
                    perm=perm,
                )
            )
            index += 1
            kernel_name = transpose_output_name

        graph.append_node(
            ir.node.math.Matmul(
                op.name + ":" + str(index), graph, [input_name, kernel_name], oup_strs
            )
        )
        return True
