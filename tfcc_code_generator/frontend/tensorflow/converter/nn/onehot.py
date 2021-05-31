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


class OneHot(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        depth_symbol = graph.get_symbol(inp_strs[1])
        if not depth_symbol.is_constant():
            return False

        on_value_symbol = graph.get_symbol(inp_strs[2])
        if not on_value_symbol.is_constant():
            return False

        off_value_symbol = graph.get_symbol(inp_strs[3])
        if not off_value_symbol.is_constant():
            return False

        axis = op.get_attr("axis")
        if axis != -1:
            return False

        one_hot_node = ir.node.nn.OneHot(
            op.name,
            graph,
            [inp_strs[0]],
            oup_strs,
            dtype=on_value_symbol.dtype,
            depth=depth_symbol.data.tolist()[0],
            on_value=on_value_symbol.data.tolist()[0],
            off_value=off_value_symbol.data.tolist()[0],
        )

        graph.append_node(one_hot_node)
        return True
