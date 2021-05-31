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

import enum
import numpy as np
from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/Slice?hl=en
class Slice(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        size_symbol = graph.get_symbol(inp_strs[2])

        if not size_symbol.is_constant():
            return False

        begins_symbol_name = graph.context.create_symbol_name(inp_strs[1])
        graph.append_node(
            ir.node.base.ToVector(
                "{}:to_tensor".format(op.name),
                graph,
                [inp_strs[1]],
                [begins_symbol_name],
            )
        )

        input_name = inp_strs[0]
        for axis, size in enumerate(size_symbol.data.tolist()):
            if size < 0:
                continue
            begin_name = graph.context.create_symbol_name(begins_symbol_name)
            graph.append_node(
                ir.node.base.At1(
                    "{}:_at_{}".format(op.name, axis),
                    graph,
                    [begins_symbol_name],
                    [begin_name],
                    idx=axis,
                )
            )
            output_name = graph.context.create_symbol_name(oup_strs[0])
            graph.append_node(
                ir.node.base.SliceV1(
                    "{}:_slice_{}".format(op.name, axis),
                    graph,
                    [input_name, begin_name],
                    [output_name],
                    axis=axis,
                    length=size,
                )
            )
            input_name = output_name

        graph.append_node(ir.node.base.Identity(op.name, graph, [input_name], oup_strs))
        return True
