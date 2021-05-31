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


class Pack(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        axis = op.get_attr("axis")
        target_shape_size = len(graph.get_symbol(inp_strs[0]).shape) + 1
        while axis < 0:
            axis += target_shape_size
        if all([graph.get_symbol(inp).is_tensor() for inp in inp_strs]):
            graph.append_node(
                ir.node.base.Stack(op.name, graph, inp_strs, oup_strs, axis=axis)
            )
        elif (
            all([graph.get_symbol(inp).is_value() for inp in inp_strs])
            and len(inp_strs) == 1
        ):
            graph.append_node(ir.node.base.ToTensor(op.name, graph, inp_strs, oup_strs))
        elif all([graph.get_symbol(inp).is_value() for inp in inp_strs]):
            create_vector_output_name = graph.context.create_symbol_name(oup_strs[0])
            inputs = []
            vec = []
            for inp in inp_strs:
                symbol = graph.get_symbol(inp)
                if symbol.is_constant():
                    vec.append([symbol.data.tolist()[0]])
                else:
                    vec.append(len(inputs))
                    inputs.append(inp)
            graph.append_node(
                ir.node.base.CreateVector(
                    op.name + ":0",
                    graph,
                    inputs,
                    [create_vector_output_name],
                    vec=vec,
                    dtype=graph.get_symbol(inp).dtype,
                )
            )
            graph.append_node(
                ir.node.base.ToTensor(
                    op.name + ":0", graph, [create_vector_output_name], oup_strs
                )
            )
        else:
            return False

        return True
