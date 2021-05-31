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


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/Split?hl=en
class Split(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        if op.get_attr("num_split") != len(oup_strs):
            return False

        axis_symbol = graph.get_symbol(inp_strs[0])
        if not axis_symbol.is_constant():
            return False
        if not axis_symbol.is_integer():
            return False

        # TODO: process arg "num_splits"
        axis = axis_symbol.data.tolist()[0]
        input_symbol = graph.get_symbol(inp_strs[1])
        while axis < 0:
            axis += len(input_symbol.shape)
        graph.append_node(
            ir.node.base.Split(op.name, graph, [inp_strs[1]], oup_strs, axis=axis)
        )
        return True
