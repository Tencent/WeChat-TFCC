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
from ir.common import create_constant_symbol
from ..converter import Converter


class Pad(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        assert len(op.inputs[1].shape) == 2 and op.inputs[1].shape[1] == 2

        pads_symbol = graph.get_symbol(op.inputs[1].name)

        # [[begin_0, end_0], [begin_1, end_1], ...] -> [begin_0, begin_1, ..., end_0, end_1]
        pads = []
        for i in [0, 1]:
            for j in range(len(pads_symbol.data)):
                pads.append(pads_symbol.data[j][i])

        # Refer to imp in onnx
        padding_infos = []
        for axis, (padding_head, padding_tail) in enumerate(
            zip(pads[: len(pads) // 2], pads[len(pads) // 2 :])
        ):
            if padding_head == 0 and padding_tail == 0:
                continue
            padding_infos.append((axis, padding_head, padding_tail))

        if len(padding_infos) == 0:
            graph.append_node(
                ir.node.base.Identity(op.name, graph, [inp_strs[0]], oup_strs)
            )
            return True

        next_input_name = inp_strs[0]
        output_names = [
            graph.context.create_symbol_name(oup_strs[0]) for _ in padding_infos[1:]
        ] + oup_strs
        for i, (output_name, (axis, padding_head, padding_tail)) in enumerate(
            zip(output_names, padding_infos)
        ):
            padding_head_name = graph.context.create_symbol_name(
                "{}_padding_head_{}".format(inp_strs[0], i)
            )
            create_constant_symbol(
                graph,
                padding_head_name,
                ir.framework.DataType.UINT32,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([padding_head], dtype=np.uint32),
            )
            padding_tail_name = graph.context.create_symbol_name(
                "{}_padding_tail_{}".format(inp_strs[0], i)
            )
            create_constant_symbol(
                graph,
                padding_tail_name,
                ir.framework.DataType.UINT32,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([padding_tail], dtype=np.uint32),
            )
            graph.append_node(
                ir.node.base.Pad(
                    "{}:{}".format(op.name, i),
                    graph,
                    [next_input_name, padding_head_name, padding_tail_name],
                    [output_name],
                    axis=axis,
                )
            )
            next_input_name = output_name
        return True
