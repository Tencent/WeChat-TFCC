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


class Loop(Converter):
    def __call__(
        self,
        node_proto: onnx.NodeProto,
        graph_proto: onnx.GraphProto,
        graph: ir.framework.Graph,
    ):
        if not self.accept(node_proto):
            return False

        attributes = self.get_attributes(node_proto)
        if not "body" in attributes:
            return False
        op_set = graph.context.frontend["op_set"]
        from frontend.onnx.frontend import graph2ir

        sub_graph, name_map = graph2ir(
            graph.context.create_graph_name(attributes["body"].name),
            attributes["body"],
            graph.model,
            op_set,
            graph,
        )
        if len(name_map) > 0:
            assert set(name_map.values()) == set(sub_graph.inputs[-len(name_map) :])

        # change iter to value
        if sub_graph.get_symbol(sub_graph.inputs[0]).is_tensor():
            sub_graph.get_symbol(
                sub_graph.inputs[0]
            ).stype = ir.framework.SymbolType.VALUE
            to_tensor_output_name = sub_graph.context.create_symbol_name(
                sub_graph.inputs[0]
            )
            sub_graph.add_node(
                0,
                ir.node.base.ToTensor(
                    "loop__input_to_tensor",
                    sub_graph,
                    [sub_graph.inputs[0]],
                    [to_tensor_output_name],
                ),
            )
            for node in sub_graph.nodes[1:]:
                if sub_graph.inputs[0] in node.input_names:
                    node.update_inputs(
                        [
                            to_tensor_output_name
                            if name == sub_graph.inputs[0]
                            else name
                            for name in node.input_names
                        ]
                    )
            sub_graph.reflash_symbols()

        cond_name = graph.context.create_symbol_name(node_proto.input[1])
        graph.append_node(
            ir.node.base.ToValue(
                node_proto.name + ":0", graph, [node_proto.input[1]], [cond_name]
            )
        )

        inputs = [cond_name] + list(node_proto.input[2:])

        name_map_reverse = {}
        for name in name_map:
            name_map_reverse[name_map[name]] = name

        if len(name_map) > 0:
            for name in sub_graph.inputs[-len(name_map) :]:
                inputs.append(name_map_reverse[name])

        if node_proto.input[0]:
            max_loop_name = graph.context.create_symbol_name(node_proto.input[0])
            graph.append_node(
                ir.node.base.ToValue(
                    node_proto.name + ":0",
                    graph,
                    [node_proto.input[0]],
                    [max_loop_name],
                )
            )
            inputs.append(max_loop_name)

        carried_count = len(sub_graph.inputs) - 2 - len(name_map)
        capture_count = len(name_map)
        scan_count = len(sub_graph.outputs) - 1 - carried_count
        scan_names = [
            graph.context.create_symbol_name(name)
            for name in node_proto.output[carried_count:]
        ]

        loop_node = ir.node.base.Loop(
            node_proto.name + ":2",
            graph,
            inputs,
            node_proto.output[:carried_count] + scan_names,
            sub_graph.name,
            carried_count,
            capture_count,
            scan_count,
        )
        graph.append_node(loop_node)
        for i, (output_name, scan_name) in enumerate(
            zip(node_proto.output[carried_count:], scan_names)
        ):
            if graph.get_symbol(scan_name).is_vector():
                graph.append_node(
                    ir.node.base.ToTensor(
                        node_proto.name + ":" + str(i + 3),
                        graph,
                        [scan_name],
                        [output_name],
                    )
                )
            else:
                graph.append_node(
                    ir.node.base.Identity(
                        node_proto.name + ":" + str(i + 3),
                        graph,
                        [scan_name],
                        [output_name],
                    )
                )

        return True

    @property
    def accept_versions(self) -> set:
        return set([1, 11])
