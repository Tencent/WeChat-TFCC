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


class Squeeze(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        axes = sorted(list(map(int, op.get_attr("squeeze_dims"))))

        input_symbol = graph.get_symbol(inp_strs[0])
        for i in range(len(axes)):
            while axes[i] < 0:
                axes[i] += len(input_symbol.shape)

        if axes == list(range(len(input_symbol.shape))):
            graph.append_node(ir.node.base.ToValue(op.name, graph, inp_strs, oup_strs))
        elif len(axes) > 0:
            graph.append_node(
                ir.node.base.Squeeze(op.name, graph, inp_strs, oup_strs, axes=axes)
            )
        elif len(axes) == 0:
            graph.append_node(ir.node.base.Identity(op.name, graph, inp_strs, oup_strs))
        else:
            raise RuntimeError("Unknow error")
        return True
