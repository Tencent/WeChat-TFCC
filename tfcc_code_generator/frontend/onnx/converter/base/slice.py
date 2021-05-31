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


class Slice(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False
        if self.since_version == 1:
            return self.process_onnx_slice_1(node_proto, graph_proto, graph)
        else:
            return self.process_onnx_slice_other(node_proto, graph_proto, graph)

    def process_onnx_slice_1(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        data_symbol = graph.get_symbol(node_proto.input[0])

        attributes = self.get_attributes(node_proto)
        if "axes" in attributes:
            axes = attributes["axes"]
        else:
            axes = list(range(len(data_symbol.shape)))
        starts = attributes["starts"]
        ends = attributes["ends"]

        data_name = node_proto.input[0]
        starts_name = graph.context.create_symbol_name(node_proto.input[0] + "_starts")
        ends_name = graph.context.create_symbol_name(node_proto.input[0] + "_starts")

        assert len(axes) == len(starts) and len(starts) == len(ends)
        for i, (axis, start, end) in enumerate(zip(axes, starts, ends)):
            while axis < 0:
                axis += len(data_symbol.shape)
            if start == 0 and end >= 2147483647 and i < len(axes) - 1:
                continue
            start_name = graph.context.create_symbol_name(starts_name + "_" + str(i))
            create_constant_symbol(
                graph,
                start_name,
                ir.framework.DataType.UINT32,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([start], dtype=np.uint32),
            )
            end_name = graph.context.create_symbol_name(ends_name + "_" + str(i))
            create_constant_symbol(
                graph,
                end_name,
                ir.framework.DataType.UINT32,
                ir.framework.SymbolType.CONSTANT_VALUE,
                [1],
                np.asarray([end], dtype=np.uint32),
            )

            if i == len(axes) - 1:
                output_name = node_proto.output[0]
            else:
                output_name = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                ir.node.base.SliceV2(
                    "{}:{}".format(node_proto.name, 3 + i * 3),
                    graph,
                    [data_name, start_name, end_name],
                    [output_name],
                    axis=axis,
                )
            )
            data_name = output_name

        return True

    def process_onnx_slice_other(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        data_symbol = graph.get_symbol(node_proto.input[0])
        axes_symbol = None
        steps_symbol = None
        if len(node_proto.input) >= 4:
            axes_symbol = graph.get_symbol(node_proto.input[3])
        if len(node_proto.input) >= 5:
            steps_symbol = graph.get_symbol(node_proto.input[4])

        if steps_symbol and not steps_symbol.is_constant():
            return False
        if axes_symbol and not axes_symbol.is_constant():
            return False

        if steps_symbol:
            for v in steps_symbol.data.tolist():
                if v != 1:
                    return False

        if not axes_symbol:
            axes = list(range(len(data_symbol.shape)))
        else:
            axes = list(axes_symbol.data.tolist())

        starts_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.ToVector(
                node_proto.name + ":0", graph, [node_proto.input[1]], [starts_name]
            )
        )
        ends_name = graph.context.create_symbol_name(node_proto.input[2])
        graph.append_node(
            ir.node.base.ToVector(
                node_proto.name + ":0", graph, [node_proto.input[2]], [ends_name]
            )
        )

        data_name = node_proto.input[0]

        # fix for slice empty tensor
        if (
            list(axes) == [0]
            and len(data_symbol.shape) == 1
            and isinstance(data_symbol.shape[0], int)
        ):
            starts_symbol = graph.get_symbol(starts_name)
            if (
                starts_symbol.is_constant()
                and starts_symbol.data.tolist()[0] >= data_symbol.shape[0]
            ):
                frontend = graph.context.frontend
                if "empty_symbol" not in frontend:
                    frontend["empty_symbol"] = set()
                frontend["empty_symbol"].add(node_proto.output[0])
                graph.context.frontend = frontend
                return True

        for i, axis in enumerate(axes):
            start_name = graph.context.create_symbol_name(starts_name + "_" + str(i))
            graph.append_node(
                ir.node.base.At1(
                    "{}:{}".format(node_proto.name, 1 + i * 3),
                    graph,
                    [starts_name],
                    [start_name],
                    idx=i,
                )
            )
            end_name = graph.context.create_symbol_name(ends_name + "_" + str(i))
            graph.append_node(
                ir.node.base.At1(
                    "{}:{}".format(node_proto.name, 2 + i * 3),
                    graph,
                    [ends_name],
                    [end_name],
                    idx=i,
                )
            )
            if i == len(axes) - 1:
                output_name = node_proto.output[0]
            else:
                output_name = graph.context.create_symbol_name(node_proto.output[0])
            graph.append_node(
                ir.node.base.SliceV2(
                    "{}:{}".format(node_proto.name, 3 + i * 3),
                    graph,
                    [data_name, start_name, end_name],
                    [output_name],
                    axis=axis,
                )
            )
            data_name = output_name

        return True

    @property
    def accept_versions(self) -> set:
        # TODO support Slice-1
        return set([1, 10, 11])
