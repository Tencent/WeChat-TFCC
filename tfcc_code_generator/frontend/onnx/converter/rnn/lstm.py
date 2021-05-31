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


class LSTM(Converter):
    def change_iofc_to_icfo(
        self, node_name: str, index, graph: ir.framework.Graph, name: str, axis: int
    ):
        i_name = graph.context.create_symbol_name(name)
        o_name = graph.context.create_symbol_name(name)
        f_name = graph.context.create_symbol_name(name)
        c_name = graph.context.create_symbol_name(name)

        graph.append_node(
            ir.node.base.Split(
                node_name + ":" + str(index),
                graph,
                [name],
                [i_name, o_name, f_name, c_name],
                axis=axis,
            )
        )

        output_name = graph.context.create_symbol_name(name)
        graph.append_node(
            ir.node.base.Concat(
                node_name + ":" + str(index + 1),
                graph,
                [i_name, c_name, f_name, o_name],
                [output_name],
                axis=axis,
            )
        )
        return output_name

    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        if "activation_alpha" in attributes:
            return False
        if "activation_beta" in attributes:
            return False
        if "activations" in attributes:
            return False
        if "clip" in attributes:
            return False
        input_forget = 0
        if "input_forget" in attributes:
            input_forget = attributes["input_forget"]
        direction = "forward"
        if "direction" in attributes:
            direction = attributes["direction"]
        if input_forget != 0 or direction != b"bidirectional":
            return False

        if "hidden_size" not in attributes:
            return False
        hidden_size = attributes["hidden_size"]

        if len(node_proto.input) != 7 or len(node_proto.output) != 3:
            return False
        if node_proto.input[4] != "":
            return False

        input_kernal_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.Transpose(
                node_proto.name + ":0",
                graph,
                [node_proto.input[1]],
                [input_kernal_name],
                perm=[0, 2, 1],
            )
        )
        state_kernal_name = graph.context.create_symbol_name(node_proto.input[2])
        graph.append_node(
            ir.node.base.Transpose(
                node_proto.name + ":1",
                graph,
                [node_proto.input[2]],
                [state_kernal_name],
                perm=[0, 2, 1],
            )
        )
        input_bias_name = graph.context.create_symbol_name(node_proto.input[3])
        state_bias_name = graph.context.create_symbol_name(node_proto.input[3])
        graph.append_node(
            ir.node.base.Split(
                node_proto.name + ":2",
                graph,
                [node_proto.input[3]],
                [input_bias_name, state_bias_name],
                axis=1,
            )
        )
        bias_name = graph.context.create_symbol_name(node_proto.input[3])
        graph.append_node(
            ir.node.math.Add(
                node_proto.name + ":3",
                graph,
                [input_bias_name, state_bias_name],
                [bias_name],
            )
        )

        forward_output_name = graph.context.create_symbol_name(node_proto.output[0])
        backward_output_name = graph.context.create_symbol_name(node_proto.output[0])
        forward_h_name = graph.context.create_symbol_name(node_proto.output[1])
        backward_h_name = graph.context.create_symbol_name(node_proto.output[1])
        forward_c_name = graph.context.create_symbol_name(node_proto.output[2])
        backward_c_name = graph.context.create_symbol_name(node_proto.output[2])

        input_kernal_name = self.change_iofc_to_icfo(
            node_proto.name, 4, graph, input_kernal_name, 2
        )
        state_kernal_name = self.change_iofc_to_icfo(
            node_proto.name, 6, graph, state_kernal_name, 2
        )
        bias_name = self.change_iofc_to_icfo(node_proto.name, 8, graph, bias_name, 1)

        inputs = [
            node_proto.input[0],
            input_kernal_name,
            state_kernal_name,
            bias_name,
            node_proto.input[5],
            node_proto.input[6],
        ]
        outputs = [
            forward_output_name,
            backward_output_name,
            forward_h_name,
            backward_h_name,
            forward_c_name,
            backward_c_name,
        ]

        graph.append_node(
            ir.node.rnn.BidirectionalLSTM(
                node_proto.name + ":9", graph, inputs, outputs, hidden_size=hidden_size
            )
        )
        graph.append_node(
            ir.node.base.Stack(
                node_proto.name + ":10",
                graph,
                [forward_output_name, backward_output_name],
                [node_proto.output[0]],
                axis=1,
            )
        )
        graph.append_node(
            ir.node.base.Stack(
                node_proto.name + ":11",
                graph,
                [forward_h_name, backward_h_name],
                [node_proto.output[1]],
                axis=0,
            )
        )
        graph.append_node(
            ir.node.base.Stack(
                node_proto.name + ":12",
                graph,
                [forward_c_name, backward_c_name],
                [node_proto.output[2]],
                axis=1,
            )
        )
        return True

    @property
    def accept_versions(self) -> set:
        return set([7])
