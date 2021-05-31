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
import onnx
import ir.node
import ir.framework
import frontend.onnx.common
from ..converter import Converter
from ir.common import create_constant_symbol


class MaxPool(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        if "auto_pad" in attributes:
            if attributes["auto_pad"] != b"NOTSET":
                return False
        if "ceil_mode" in attributes:
            if attributes["ceil_mode"] != 0:
                return False
        if "dilations" in attributes:
            if not all([v == 1 for v in attributes["dilations"]]):
                return False
        if "storage_order" in attributes:
            if attributes["storage_order"] != 0:
                return False
        if "kernel_shape" not in attributes:
            return False
        symbol = graph.get_symbol(node_proto.input[0])

        kernels = attributes["kernel_shape"]
        pads = [0 for _ in range(len(symbol.shape) - 2)] * 2
        strides = [1 for _ in range(len(symbol.shape) - 2)]
        individual_pad = False
        if "pads" in attributes:
            if len(attributes["pads"]) != len(pads):
                return False
            if (
                attributes["pads"][: len(attributes["pads"]) // 2]
                != attributes["pads"][len(attributes["pads"]) // 2 :]
            ):
                individual_pad = True
            else:
                pads = attributes["pads"]

        if "strides" in attributes:
            if len(attributes["strides"]) != len(strides):
                return False
            strides = attributes["strides"]

        kernels = kernels[: len(symbol.shape) - 2]
        pads = pads[: len(symbol.shape) - 2]
        strides = strides[: len(symbol.shape) - 2]

        input_name = node_proto.input[0]
        if len(symbol.shape) == 3:
            data_format = ir.node.nn.MaxPool.DataFormat.NCW
        else:
            data_format = ir.node.nn.MaxPool.DataFormat.NCHW
        if individual_pad:
            for i, (padding_head, padding_tail) in enumerate(
                zip(
                    attributes["pads"][: len(attributes["pads"]) // 2],
                    attributes["pads"][len(attributes["pads"]) // 2 :],
                )
            ):
                if padding_head == 0 and padding_tail == 0:
                    continue
                axis = i + 2
                padding_head_name = graph.context.create_symbol_name(
                    input_name + "_padding_head_" + str(i)
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
                    input_name + "_padding_tail_" + str(i)
                )
                create_constant_symbol(
                    graph,
                    padding_tail_name,
                    ir.framework.DataType.UINT32,
                    ir.framework.SymbolType.CONSTANT_VALUE,
                    [1],
                    np.asarray([padding_tail], dtype=np.uint32),
                )
                padding_output_name = graph.context.create_symbol_name(
                    node_proto.output[0]
                )
                graph.append_node(
                    ir.node.base.Pad(
                        node_proto.name + ":padding_" + str(i),
                        graph,
                        [input_name, padding_head_name, padding_tail_name],
                        [padding_output_name],
                        axis=axis,
                    )
                )
                input_name = padding_output_name

        graph.append_node(
            ir.node.nn.MaxPool(
                node_proto.name,
                graph,
                [input_name],
                node_proto.output,
                kernels=kernels,
                pads=pads,
                strides=strides,
                data_format=data_format,
            )
        )
        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 8, 10, 11, 12])
