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


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/MatrixBandPart?hl=en
class MatrixBandPart(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        num_lower_symbol = graph.get_symbol(inp_strs[1])
        num_upper_symbol = graph.get_symbol(inp_strs[2])

        if not num_lower_symbol.is_constant() or not num_upper_symbol.is_constant():
            return False

        if num_lower_symbol.data.size > 1 or num_upper_symbol.data.size > 1:
            return False

        if (
            num_lower_symbol.data.tolist()[0] < 0
            and num_upper_symbol.data.tolist()[0] < 0
        ):
            graph.append_node(
                ir.node.base.Identity(op.name, graph, [inp_strs[0]], oup_strs)
            )
        elif (
            num_lower_symbol.data.tolist()[0] < 0
            and num_upper_symbol.data.tolist()[0] >= 0
        ):
            k = num_upper_symbol.data.tolist()[0]
            graph.append_node(
                ir.node.base.Tril(op.name, graph, [inp_strs[0]], oup_strs, k=k)
            )
        elif (
            num_lower_symbol.data.tolist()[0] >= 0
            and num_upper_symbol.data.tolist()[0] < 0
        ):
            k = -num_lower_symbol.data.tolist()[0]
            graph.append_node(
                ir.node.base.Triu(op.name, graph, [inp_strs[0]], oup_strs, k=k)
            )
        else:
            k_tril = num_upper_symbol.data.tolist()[0]
            k_triu = -num_lower_symbol.data.tolist()[0]

            tmp_output_name = graph.context.create_symbol_name(inp_strs[0])
            graph.append_node(
                ir.node.base.Tril(
                    op.name + ":0", graph, [inp_strs[0]], [tmp_output_name], k=k_tril
                )
            )
            graph.append_node(
                ir.node.base.Triu(
                    op.name + ":1", graph, [tmp_output_name], oup_strs, k=k_triu
                )
            )
        return True
