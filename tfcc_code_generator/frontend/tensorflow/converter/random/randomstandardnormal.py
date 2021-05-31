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

from typing import cast
from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/RandomStandardNormal?hl=en
class RandomStandardNormal(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]
        if len(inp_strs) < 1:
            return False

        dtype = op.get_attr("dtype")

        cast_output_name = graph.context.create_symbol_name(inp_strs[0])
        graph.append_node(
            ir.node.base.Cast(
                op.name + ":0",
                graph,
                inp_strs,
                [cast_output_name],
                dtype=ir.framework.DataType.UINT32,
            )
        )
        to_vector_output_name = graph.context.create_symbol_name(inp_strs[0])
        graph.append_node(
            ir.node.base.ToVector(
                op.name + ":1", graph, [cast_output_name], [to_vector_output_name]
            )
        )
        graph.append_node(
            ir.node.random.NormalLike(
                op.name + ":2",
                graph,
                [to_vector_output_name],
                oup_strs,
                dtype=dtype,
                mean=0.0,
                scale=1.0,
            )
        )
        return True
