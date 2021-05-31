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


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/StridedSlice?hl=en
class StridedSlice(Converter):
    def mask_to_axes(self, mask):
        axes = set()
        for i in range(int(mask ** 0.5) + 1):
            if 1 << i & mask:
                axes.add(i)
        return axes

    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        # TODO no support
        if op.get_attr("ellipsis_mask") != 0:
            return False

        begin_ignore_axes = self.mask_to_axes(op.get_attr("begin_mask"))
        end_ignore_axes = self.mask_to_axes(op.get_attr("end_mask"))
        new_axes = self.mask_to_axes(op.get_attr("new_axis_mask"))
        shrink_axes = self.mask_to_axes(op.get_attr("shrink_axis_mask"))
        if shrink_axes and new_axes:
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        input_symbol = graph.get_symbol(inp_strs[0])
        begin_symbol = graph.get_symbol(inp_strs[1])
        end_symbol = graph.get_symbol(inp_strs[2])
        strides_symbol = graph.get_symbol(inp_strs[3])
        # TODO no support
        if set(strides_symbol.data.tolist()) != set([1]):
            return False

        assert begin_symbol.dtype == end_symbol.dtype
        begin_vector_name = graph.context.create_symbol_name(begin_symbol.name)
        graph.append_node(
            ir.node.base.ToVector(
                op.name + ":0", graph, [begin_symbol.name], [begin_vector_name]
            )
        )
        end_vector_name = graph.context.create_symbol_name(end_symbol.name)
        graph.append_node(
            ir.node.base.ToVector(
                op.name + ":1", graph, [end_symbol.name], [end_vector_name]
            )
        )

        zero_name = graph.context.create_symbol_name("zero")
        infinity_name = graph.context.create_symbol_name("infinity")

        create_constant_symbol(
            graph,
            zero_name,
            begin_symbol.dtype,
            ir.framework.SymbolType.CONSTANT_VALUE,
            [1],
            np.asarray([0], dtype=begin_symbol.dtype.numpy_dtype),
        )
        max_value = np.iinfo(end_symbol.dtype.numpy_dtype).max
        create_constant_symbol(
            graph,
            infinity_name,
            end_symbol.dtype,
            ir.framework.SymbolType.CONSTANT_VALUE,
            [1],
            np.asarray([max_value], dtype=end_symbol.dtype.numpy_dtype),
        )

        input_name = inp_strs[0]
        if new_axes:
            output_name = graph.context.create_symbol_name(input_name)
            graph.append_node(
                ir.node.base.Unsqueeze(
                    op.name + ":2", graph, [input_name], [output_name], axes=new_axes
                )
            )
            input_name = output_name

        for i in range(len(input_symbol.shape)):
            if i in new_axes:
                continue
            if i in begin_ignore_axes or (
                isinstance(graph.get_symbol(begin_vector_name).shape[0], int)
                and i >= graph.get_symbol(begin_vector_name).shape[0]
            ):
                begin_name = zero_name
            else:
                begin_name = graph.context.create_symbol_name(begin_symbol.name)
                graph.append_node(
                    ir.node.base.At1(
                        op.name + ":" + str(i * 3 + 3),
                        graph,
                        [begin_vector_name],
                        [begin_name],
                        idx=i,
                    )
                )
            if i in end_ignore_axes or (
                isinstance(graph.get_symbol(end_vector_name).shape[0], int)
                and i >= graph.get_symbol(end_vector_name).shape[0]
            ):
                end_name = infinity_name
            else:
                end_name = graph.context.create_symbol_name(end_symbol.name)
                graph.append_node(
                    ir.node.base.At1(
                        op.name + ":" + str(i * 3 + 4),
                        graph,
                        [end_vector_name],
                        [end_name],
                        idx=i,
                    )
                )
            if (
                graph.get_symbol(begin_name).is_constant()
                and graph.get_symbol(end_name).is_constant()
            ):
                if graph.get_symbol(begin_name).data.tolist() == [
                    0
                ] and graph.get_symbol(end_name).data.tolist() == [-1]:
                    continue
                # fix [-1:0] => [-1:]
                if graph.get_symbol(begin_name).data.tolist() == [
                    -1
                ] and graph.get_symbol(end_name).data.tolist() == [0]:
                    end_name = infinity_name
            output_name = graph.context.create_symbol_name(oup_strs[0])
            graph.append_node(
                ir.node.base.SliceV2(
                    op.name + ":" + str(i * 3 + 5),
                    graph,
                    [input_name, begin_name, end_name],
                    [output_name],
                    axis=i,
                )
            )
            input_name = output_name

        if len(shrink_axes) != 0:
            if len(graph.get_symbol(input_name).shape) == 1:
                assert shrink_axes == set([0])
                graph.append_node(
                    ir.node.base.ToValue(op.name, graph, [input_name], [oup_strs[0]])
                )
            else:
                graph.append_node(
                    ir.node.base.Squeeze(
                        op.name, graph, [input_name], [oup_strs[0]], axes=shrink_axes
                    )
                )
        else:
            graph.append_node(
                ir.node.base.Identity(op.name, graph, [input_name], [oup_strs[0]])
            )

        return True
