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


class GatherV2(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        batch_dims = op.get_attr("batch_dims")
        if batch_dims != 0:
            # TODO no support
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        input_symbol = graph.get_symbol(inp_strs[0])
        indices_symbol = graph.get_symbol(inp_strs[1])
        axis_symbol = graph.get_symbol(inp_strs[2])

        axis = 0
        if len(axis_symbol.data) > 0:
            axis = list(map(int, axis_symbol.data))[0]
        while axis < 0:
            axis = axis + len(input_symbol.shape)

        if input_symbol.is_tensor() and indices_symbol.is_tensor():
            graph.append_node(
                ir.node.nn.Gather(
                    op.name, graph, [inp_strs[0], inp_strs[1]], oup_strs, axis=axis
                )
            )
            return True
        elif (
            input_symbol.is_tensor()
            and indices_symbol.is_value()
            and len(input_symbol.shape) == 1
        ):
            to_vector_output_name = graph.context.create_symbol_name(oup_strs[0])
            graph.append_node(
                ir.node.base.ToVector(
                    op.name + ":0", graph, [inp_strs[0]], [to_vector_output_name]
                )
            )
            graph.append_node(
                ir.node.base.At(
                    op.name + ":1",
                    graph,
                    [to_vector_output_name, inp_strs[1]],
                    oup_strs,
                )
            )
            return True
        elif (
            input_symbol.is_tensor()
            and indices_symbol.is_value()
            and len(indices_symbol.shape) > 1
        ):
            slice_output_name = graph.context.create_symbol_name(oup_strs[0])
            graph.append_node(
                ir.node.base.SliceV1(
                    op.name + ":0",
                    graph,
                    [inp_strs[0], inp_strs[1]],
                    [slice_output_name],
                    axis=axis,
                    length=1,
                )
            )
            graph.append_node(
                ir.node.base.Squeeze(
                    op.name + ":1", graph, [slice_output_name], oup_strs, axes=[axis]
                )
            )
            return True
        else:
            return False
