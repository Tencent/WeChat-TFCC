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


class Pad(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        if self.since_version >= 11:
            return self.process_new_version(node_proto, graph_proto, graph)

        attributes = self.get_attributes(node_proto)
        if "mode" in attributes and attributes["mode"] != b"constant":
            return False
        if "value" in attributes and attributes["value"] != 0:
            return False

        pads = attributes["pads"]
        if len(pads) % 2 != 0:
            raise RuntimeError("Unknow attribute pads")
        padding_infos = []
        for axis, (padding_head, padding_tail) in enumerate(
            zip(pads[: len(pads) // 2], pads[len(pads) // 2 :])
        ):
            if padding_head == 0 and padding_tail == 0:
                continue
            padding_infos.append((axis, padding_head, padding_tail))

        if len(padding_infos) == 0:
            graph.append_node(
                ir.node.base.Identity(
                    node_proto.name, graph, node_proto.input, node_proto.output
                )
            )
            return True

        next_input_name = node_proto.input[0]
        output_names = [
            graph.context.create_symbol_name(node_proto.output[0])
            for _ in padding_infos[1:]
        ] + [node_proto.output[0]]
        for i, (output_name, (axis, padding_head, padding_tail)) in enumerate(
            zip(output_names, padding_infos)
        ):
            padding_head_name = graph.context.create_symbol_name(
                node_proto.input[0] + "_padding_head" + "_" + str(i)
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
                node_proto.input[0] + "_padding_tail" + "_" + str(i)
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
                    node_proto.name + ":" + str(i),
                    graph,
                    [next_input_name, padding_head_name, padding_tail_name],
                    [output_name],
                    axis=axis,
                )
            )
            next_input_name = output_name
        return True

    def process_new_version(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        attributes = self.get_attributes(node_proto)
        if "mode" in attributes and attributes["mode"] != b"constant":
            return False
        if len(node_proto.input) == 3:
            value_symbol = graph.get_symbol(node_proto.input[2])
            if not value_symbol.is_constant():
                return False
            if value_symbol.data.tolist()[0] != 0:
                return False
        to_vector_output_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.ToVector(
                node_proto.name + ":0",
                graph,
                [node_proto.input[1]],
                [to_vector_output_name],
            )
        )

        input_symbol = graph.get_symbol(node_proto.input[0])

        next_input_name = node_proto.input[0]
        output_names = [
            graph.context.create_symbol_name(node_proto.output[0])
            for _ in range(len(input_symbol.shape) - 1)
        ] + [node_proto.output[0]]
        for i, output_name in enumerate(output_names):
            padding_head_name = graph.context.create_symbol_name(
                node_proto.input[1] + "_padding_head"
            )
            graph.append_node(
                ir.node.base.At1(
                    node_proto.name + ":" + str(i * 3 + 1),
                    graph,
                    [to_vector_output_name],
                    [padding_head_name],
                    idx=i,
                )
            )
            padding_tail_name = graph.context.create_symbol_name(
                node_proto.input[0] + "_padding_tail" + "_" + str(i)
            )
            graph.append_node(
                ir.node.base.At1(
                    node_proto.name + ":" + str(i * 3 + 2),
                    graph,
                    [to_vector_output_name],
                    [padding_tail_name],
                    idx=i + len(input_symbol.shape),
                )
            )
            graph.append_node(
                ir.node.base.Pad(
                    node_proto.name + ":" + str(i * 3 + 2),
                    graph,
                    [next_input_name, padding_head_name, padding_tail_name],
                    [output_name],
                    axis=i,
                )
            )
            next_input_name = output_name
        return True

    @property
    def accept_versions(self) -> set:
        return set([2, 11])
