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


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/Conv2D?hl=en
class Conv2D(Converter):
    def get_padding_size(self, input_size, stride_size, kernel_size):
        if stride_size == 1:
            return kernel_size - 1
        else:
            if not isinstance(input_size, int):
                return -1
            output_size = (input_size + stride_size - 1) // 2
            padding = (output_size - 1) * stride_size + kernel_size - input_size
            padding = max(0, padding)
            return padding

    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        assert len(inp_strs) == 2
        assert len(oup_strs) == 1

        input_symbol = graph.get_symbol(inp_strs[0])
        kernel_symbol = graph.get_symbol(inp_strs[1])

        data_format = op.get_attr("data_format")
        strides = op.get_attr("strides")
        padding = op.get_attr("padding")
        dilations = op.get_attr("dilations")

        if data_format == b"NHWC":
            n, h, w, c = input_symbol.shape
            nchw = False
        elif data_format == b"NCHW":
            n, c, h, w = input_symbol.shape
            nchw = True
        else:
            return False

        kernel_height = kernel_symbol.shape[0]
        kernel_width = kernel_symbol.shape[1]

        if len(strides) != 4:
            return False
        if nchw:
            if strides[0] != 1 or strides[1] != 1:
                return False
            stride_height = strides[2]
            stride_width = strides[3]
        else:
            if strides[0] != 1 or strides[3] != 1:
                return False
            stride_height = strides[1]
            stride_width = strides[2]

        if len(dilations) != 4:
            return False
        if nchw:
            if dilations[0] != 1 or dilations[1] != 1:
                return False
            dilation_height = dilations[2]
            dilation_width = dilations[3]
        else:
            if dilations[0] != 1 or dilations[3] != 1:
                return False
            dilation_height = dilations[1]
            dilation_width = dilations[2]

        real_input_name = inp_strs[0]

        if padding == b"VALID":
            padding_height = 0
            padding_width = 0
        elif padding == b"SAME":
            if dilation_height != 1 or dilation_width != 1:
                return False
            if not isinstance(kernel_height, int) or not isinstance(kernel_width, int):
                return False

            total_padding_height = self.get_padding_size(
                h, stride_height, kernel_height
            )
            total_padding_width = self.get_padding_size(w, stride_width, kernel_width)
            if total_padding_height < 0 or total_padding_width < 0:
                return False
            if total_padding_height % 2 == 0:
                padding_height = total_padding_height // 2
            else:
                padding_output_name = graph.context.create_symbol_name(real_input_name)
                padding_head_name = graph.context.create_symbol_name(
                    "{}_padding_head".format(inp_strs[0])
                )
                create_constant_symbol(
                    graph,
                    padding_head_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([total_padding_height // 2], dtype=np.uint32),
                )
                padding_tail_name = graph.context.create_symbol_name(
                    "{}_padding_tail".format(inp_strs[0])
                )
                create_constant_symbol(
                    graph,
                    padding_tail_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([total_padding_height // 2 + 1], dtype=np.uint32),
                )
                if data_format == b"NHWC":
                    axis = 1
                else:
                    axis = 2
                graph.append_node(
                    ir.node.base.Pad(
                        op.name + ":0",
                        graph,
                        [real_input_name, padding_head_name, padding_tail_name],
                        [padding_output_name],
                        axis=axis,
                    )
                )
                real_input_name = padding_output_name
                padding_height = 0
            if total_padding_width % 2 == 0:
                padding_width = total_padding_width // 2
            else:
                padding_output_name = graph.context.create_symbol_name(real_input_name)
                padding_head_name = graph.context.create_symbol_name(
                    "{}_padding_head".format(inp_strs[0])
                )
                create_constant_symbol(
                    graph,
                    padding_head_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([total_padding_width // 2], dtype=np.uint32),
                )
                padding_tail_name = graph.context.create_symbol_name(
                    "{}_padding_tail".format(inp_strs[0])
                )
                create_constant_symbol(
                    graph,
                    padding_tail_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([total_padding_width // 2 + 1], dtype=np.uint32),
                )
                if data_format == b"NHWC":
                    axis = 2
                else:
                    axis = 3
                graph.append_node(
                    ir.node.base.Pad(
                        op.name + ":1",
                        graph,
                        [real_input_name, padding_head_name, padding_tail_name],
                        [padding_output_name],
                        axis=axis,
                    )
                )
                real_input_name = padding_output_name
                padding_width = 0
        else:
            return False

        kernel_transpose_output_name = graph.context.create_symbol_name(inp_strs[1])
        graph.append_node(
            ir.node.base.Transpose(
                op.name + ":2",
                graph,
                [inp_strs[1]],
                [kernel_transpose_output_name],
                perm=[3, 2, 0, 1],
            )
        )
        attrs = {
            "padding_height": padding_height,
            "padding_width": padding_width,
            "stride_height": stride_height,
            "stride_width": stride_width,
            "dilate_height": dilation_height,
            "dilate_width": dilation_width,
            "group": 1,
            "nchw": nchw,
        }
        graph.append_node(
            ir.node.nn.Conv(
                op.name + ":3",
                graph,
                [real_input_name, kernel_transpose_output_name],
                oup_strs,
                **attrs
            )
        )
        return True
