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


class MaxPool(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        inp_strs = [inp.name for inp in op.inputs]
        oup_strs = [oup.name for oup in op.outputs]

        input_symbol = graph.get_symbol(inp_strs[0])
        data_format = op.get_attr("data_format")
        ksize = op.get_attr("ksize")
        padding = op.get_attr("padding")
        strides = op.get_attr("strides")

        if len(input_symbol.shape) != 4:
            return False
        if data_format == b"NHWC":
            nchw = False
        elif data_format == b"NCHW":
            nchw = True
        else:
            return False

        if len(ksize) != 4:
            return False
        if nchw:
            if ksize[0] != 1 or ksize[1] != 1:
                return False
            kernel_height = ksize[2]
            kernel_width = ksize[3]
        else:
            if ksize[0] != 1 or ksize[3] != 1:
                return False
            kernel_height = ksize[1]
            kernel_width = ksize[2]

        if padding != b"VALID":
            return False
        padding_height = 0
        padding_width = 0

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

        attrs = {
            "kernels": [kernel_height, kernel_width],
            "pads": [padding_height, padding_width],
            "strides": [stride_height, stride_width],
            "data_format": ir.node.nn.MaxPool.DataFormat.NCHW
            if nchw
            else ir.node.nn.MaxPool.DataFormat.NHWC,
        }
        graph.append_node(
            ir.node.nn.MaxPool(op.name, graph, inp_strs, oup_strs, **attrs)
        )
        return True
