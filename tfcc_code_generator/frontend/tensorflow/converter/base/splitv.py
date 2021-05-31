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

import numpy as np
from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


# implement for https://www.tensorflow.org/api_docs/python/tf/raw_ops/SplitV?hl=en
class SplitV(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        num_split = op.get_attr("num_split")

        assert len(inp_strs) == 3
        assert num_split == len(oup_strs)

        axis_symbol = graph.get_symbol(inp_strs[2])
        if not axis_symbol.is_constant():
            return False

        axis = axis_symbol.data.tolist()[0]

        split_symbol = graph.get_symbol(inp_strs[1])
        if not split_symbol.is_constant():
            return False
        if len(set(split_symbol.data.tolist())) != 1:
            return False
        assert split_symbol.data.size == num_split

        input_symbol = graph.get_symbol(inp_strs[0])
        if axis < 0:
            axis += len(input_symbol.shape)

        graph.append_node(
            ir.node.base.Split(op.name, graph, [inp_strs[0]], oup_strs, axis=axis)
        )
        return True
