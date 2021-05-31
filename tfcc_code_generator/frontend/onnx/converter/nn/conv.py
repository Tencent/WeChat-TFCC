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

import onnx
import ir.node
import ir.framework
import frontend.onnx.common
from ..converter import Converter
from ir.common import create_constant_symbol


class Conv(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        dilate_height = 1
        dilate_width = 1
        if "dilations" in attributes:
            if attributes["dilations"] != [1, 1]:
                dilate_height = attributes["dilations"][0]
                dilate_width = attributes["dilations"][1]
        if "group" in attributes:
            if attributes["group"] != 1:
                return False
        if "kernel_shape" in attributes:
            if len(attributes["kernel_shape"]) != 2:
                return False
            kernel = graph.get_symbol(node_proto.input[1])
            if (
                isinstance(kernel.shape[2], int)
                and attributes["kernel_shape"][0] != kernel.shape[2]
            ):
                return False
            if (
                isinstance(kernel.shape[3], int)
                and attributes["kernel_shape"][1] != kernel.shape[3]
            ):
                return False
        padding_height = 0
        padding_width = 0
        stride_height = 1
        stride_width = 1
        individual_pad = False
        if "pads" in attributes:
            if len(attributes["pads"]) != 4:
                return False
            if attributes["pads"][:2] != attributes["pads"][2:]:
                individual_pad = True
            else:
                padding_height = attributes["pads"][0]
                padding_width = attributes["pads"][1]

        if "strides" in attributes:
            if len(attributes["strides"]) != 2:
                return False
            stride_height = attributes["strides"][0]
            stride_width = attributes["strides"][1]

        input_name = node_proto.input[0]
        kernel_name = node_proto.input[1]
        bias_name = None
        if len(node_proto.input) > 2:
            bias_name = node_proto.input[2]

        if individual_pad:
            for i, (padding_head, padding_tail) in enumerate(
                zip(attributes["pads"][:2], attributes["pads"][2:])
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

        attrs = {
            "padding_height": padding_height,
            "padding_width": padding_width,
            "stride_height": stride_height,
            "stride_width": stride_width,
            "dilate_height": dilate_height,
            "dilate_width": dilate_width,
            "group": 1,
            "nchw": True,
        }
        if not bias_name:
            graph.append_node(
                ir.node.nn.Conv(
                    node_proto.name,
                    graph,
                    [input_name, kernel_name],
                    node_proto.output,
                    **attrs
                )
            )
        else:
            conv_output_name = graph.context.create_symbol_name(node_proto.output[0])
            bias_symbol = graph.get_symbol(bias_name)
            if len(bias_symbol.shape) != 1:
                return False
            unsqueeze_output_name = graph.context.create_symbol_name(bias_name)
            graph.append_node(
                ir.node.base.Unsqueeze(
                    node_proto.name + ":0",
                    graph,
                    [bias_name],
                    [unsqueeze_output_name],
                    axes=[0, 2, 3],
                )
            )
            graph.append_node(
                ir.node.nn.Conv(
                    node_proto.name + ":1",
                    graph,
                    [input_name, kernel_name],
                    [conv_output_name],
                    **attrs
                )
            )
            graph.append_node(
                ir.node.math.Add(
                    node_proto.name + ":2",
                    graph,
                    [conv_output_name, unsqueeze_output_name],
                    node_proto.output,
                )
            )
        return True

    @property
    def accept_versions(self) -> set:
        # TODO Support dilations, group
        return set([1, 11])
