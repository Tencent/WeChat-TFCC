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

import tensorflow as tf
from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


class ArgMax(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        if not graph.get_symbol(inp_strs[1]).is_constant():
            return False

        symbol = graph.get_symbol(inp_strs[0])
        axis = graph.get_symbol(inp_strs[1]).data.tolist()[0]
        while axis < 0:
            axis += len(symbol.shape)

        argmax_output_name = graph.context.create_symbol_name(oup_strs[0])
        graph.append_node(
            ir.node.math.ArgMax(
                op.name + ":0", graph, [inp_strs[0]], [argmax_output_name], axis=axis
            )
        )

        squeeze_output_name = graph.context.create_symbol_name(oup_strs[0])
        if len(symbol.shape) == 1:
            graph.append_node(
                ir.node.base.ToValue(
                    op.name + ":1", graph, [argmax_output_name], [squeeze_output_name]
                )
            )
        else:
            graph.append_node(
                ir.node.base.Squeeze(
                    op.name + ":1",
                    graph,
                    [argmax_output_name],
                    [squeeze_output_name],
                    axes=[axis],
                )
            )

        if op.get_attr("output_type") == tf.int32:
            graph.append_node(
                ir.node.base.Cast(
                    op.name + ":2",
                    graph,
                    [squeeze_output_name],
                    oup_strs,
                    dtype=ir.framework.DataType.INT32,
                )
            )
        else:
            graph.append_node(
                ir.node.base.Identity(
                    op.name + ":2", graph, [squeeze_output_name], oup_strs
                )
            )

        return True
