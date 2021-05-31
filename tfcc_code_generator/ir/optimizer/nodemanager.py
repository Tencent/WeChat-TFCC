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

import typing
import ir.framework
import ir.node


class NodeManager(object):
    def __init__(self, graph: ir.framework.Graph):
        name_map = {}
        type_map = {}
        for node in graph.nodes:
            if node.__class__ not in type_map:
                type_map[node.__class__] = []
            type_map[node.__class__].append(node)
            for name in node.output_names:
                assert name not in name_map
                name_map[name] = node

        self._graph = graph
        self._name_map = name_map
        self._type_map = type_map

    @property
    def graph(self):
        return self._graph

    @property
    def name_map(self) -> typing.Dict[str, ir.node.Node]:
        return self._name_map

    @property
    def type_map(self) -> typing.Dict[type, typing.List[ir.node.Node]]:
        return self._type_map

    def remove_node(self, node: ir.node.Node):
        assert node in self.type_map[node.__class__]
        assert all(
            [
                name in self.name_map and self.name_map[name] == node
                for name in node.output_names
            ]
        )
        self.type_map[node.__class__].remove(node)
        for name in node.output_names:
            self.name_map.pop(name)
            self.graph.remove_symbol(name)
        self.graph.remove_node(node)

    def unrefer_node_output(self, node: ir.node.Node, idx: int):
        self._name_map.pop(node.output_names[idx])
        self.graph.remove_symbol(node.output_names[idx])
        output_names = node.output_names
        output_names[idx] = self.graph.context.create_symbol_name(
            output_names[idx] + "_unref"
        )
        self._name_map[output_names[idx]] = node
        node.update_outputs(output_names)

    def add_node(self, idx: int, node: ir.node.Node):
        if not node.__class__ in self._type_map:
            self._type_map[node.__class__] = []
        self._type_map[node.__class__].append(node)
        for name in node.output_names:
            self._name_map[name] = node

        self.graph.add_node(idx, node)
