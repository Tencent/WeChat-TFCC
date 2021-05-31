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


class RFFT(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]
        if len(inp_strs) != 2 or len(oup_strs) != 1:
            return False

        if (
            not graph.get_symbol(inp_strs[1]).is_constant()
            or not graph.get_symbol(inp_strs[1]).is_integer()
            or graph.get_symbol(inp_strs[1]).shape != [1]
        ):
            return False

        length = graph.get_symbol(inp_strs[1]).data.tolist()[0]

        graph.append_node(
            ir.node.signal.RFFT(op.name, graph, [inp_strs[0]], oup_strs, length=length)
        )
        return True
