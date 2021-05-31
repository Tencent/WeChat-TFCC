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


class ExpandDims(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        input_symbol = graph.get_symbol(inp_strs[0])
        axes_symbol = graph.get_symbol(inp_strs[1])

        axes = sorted(set(axes_symbol.data))
        target_shape_len = len(input_symbol.shape) + len(axes)
        for i, axis in enumerate(axes):
            while axis < 0:
                axis += target_shape_len
            axes[i] = axis

        if input_symbol.is_value():
            assert axes == list(range(len(axes)))
            if len(axes) == 1:
                graph.append_node(
                    ir.node.base.ToTensor(op.name, graph, [inp_strs[0]], oup_strs)
                )
            else:
                to_tensor_output = graph.context.create_symbol_name(oup_strs[0])
                graph.append_node(
                    ir.node.base.ToTensor(
                        op.name + ":0", graph, [inp_strs[0]], [to_tensor_output]
                    )
                )
                graph.append_node(
                    ir.node.base.Unsqueeze(
                        op.name + ":1",
                        graph,
                        [to_tensor_output],
                        oup_strs,
                        axes=range(len(axes) - 1),
                    )
                )
            return True
        else:
            graph.append_node(
                ir.node.base.Unsqueeze(
                    op.name, graph, [inp_strs[0]], oup_strs, axes=axes
                )
            )
            return True
