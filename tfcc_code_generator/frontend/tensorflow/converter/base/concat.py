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


class Concat(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        ignored_inp_count = 0
        inp_strs = []
        for inp in op.inputs:
            if (
                "empty_symbol" in graph.context.frontend
                and inp.name in graph.context.frontend["empty_symbol"]
            ):
                ignored_inp_count += 1
            else:
                inp_strs.append(inp.name)

        oup_strs = [oup.name for oup in op.outputs]

        N = op.get_attr("N") - ignored_inp_count
        assert len(inp_strs) == N + 1

        axis_symbol = graph.get_symbol(inp_strs[N])
        axis = list(map(int, axis_symbol.data))[0]

        input_symbol = graph.get_symbol(inp_strs[0])
        while axis < 0:
            axis = axis + len(input_symbol.shape)

        if N > 1:
            graph.append_node(
                ir.node.base.Concat(op.name, graph, inp_strs[:-1], oup_strs, axis=axis)
            )
        else:
            graph.append_node(
                ir.node.base.Identity(op.name, graph, inp_strs[:-1], oup_strs)
            )
        return True
