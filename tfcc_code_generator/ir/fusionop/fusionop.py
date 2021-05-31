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

import logging
import ir
from ir.node import Node
from ir.common import get_broadcast_shape
from ir.framework.graph import Graph
from .common import get_fusionop_maps


class FusionOp(object):
    def __init__(self):
        self._node_map = get_fusionop_maps()

    def process(self, graph: ir.framework.Graph):
        # count_map = {}
        # for node in graph.nodes:
        #     if node.__class__ not in count_map:
        #         count_map[node.__class__] = 1
        #     else:
        #         count_map[node.__class__] = count_map[node.__class__] + 1
        # for kv in count_map.items():
        #     print(kv)
        # print("nodes size:", len(graph.nodes))
        logging.info("Before fusionop, graph has {} node".format(len(graph.nodes)))
        changed = True
        while changed:
            input_node_map = {}
            output_node_map = {}
            for node in graph.nodes:
                for out in node.output_names:
                    output_node_map[out] = node
                for inp in node.input_names:
                    if inp not in input_node_map:
                        input_node_map[inp] = set()
                    input_node_map[inp].add(node)

            changed = self.runonce(graph, input_node_map, output_node_map)
            # print("nodes size:", len(graph.nodes))
        logging.info("After fusionop, graph has {} node".format(len(graph.nodes)))
        return changed

    def dfs(self, input_node_map, output_node_map, allnodes, nodes, inputs, node):
        for inp in reversed(node.input_names):
            if inp not in output_node_map:
                inputs.append(inp)
                continue
            node2 = output_node_map[inp]
            if (node2.__class__ not in self._node_map) or (
                len(input_node_map[inp]) > 1
            ):
                inputs.append(inp)
            else:
                allnodes.append(node2)
                if node2 not in nodes:
                    nodes.add(node2)
                    self.dfs(
                        input_node_map, output_node_map, allnodes, nodes, inputs, node2
                    )

    def dfs_for_rpn(self, input_node_map, output_node_map, newnodes, rpn, node):
        for inp in reversed(node.input_names):
            if inp not in output_node_map:
                rpn.append(inp)
                continue
            node2 = output_node_map[inp]
            if (node2.__class__ not in self._node_map) or (
                len(input_node_map[inp]) > 1
            ):
                rpn.append(inp)
            else:
                for idx, n in enumerate(newnodes):
                    if n[0] == node2:
                        rpn.append(n[0])
                        if n[1]:
                            self.dfs_for_rpn(
                                input_node_map, output_node_map, newnodes, rpn, n[0]
                            )
                        del newnodes[idx]
                        break

    def get_result_shape_and_broadcast_marks(self, graph: ir.framework.Graph, inputs):
        symbols = [graph.get_symbol(name) for name in inputs]
        result_shape = get_broadcast_shape(
            [inp.shape for inp in symbols], graph.context
        )
        for s in result_shape:
            if isinstance(s, str):
                return None, None

        broadcast_marks = []
        for inp in symbols:
            assert len(result_shape) >= len(inp.shape)
            l = len(result_shape) - len(inp.shape)
            for _ in range(l):
                broadcast_marks.append(True)
            for s in inp.shape:
                broadcast_marks.append(s <= 1)

        return result_shape, broadcast_marks

    def gen_new_graph(
        self,
        graph: ir.framework.Graph,
        input_node_map,
        output_node_map,
        allnodes,
        nodes,
        node,
        inputs,
    ):
        inputs.reverse()
        inputs_map = {}
        index = 0
        newinputs = []
        for i in range(len(inputs)):
            if inputs[i] not in inputs_map:
                inputs_map[inputs[i]] = index
                newinputs.append(inputs[i])
                index = index + 1
        inputs = newinputs
        outputs = node.output_names

        # handle same node
        allnodes.reverse()
        newnodes = set()
        newallnodes = []
        for n in allnodes:
            if n not in newnodes:
                newnodes.add(n)
                newallnodes.append((n, True))
            else:
                newallnodes.append((n, False))
        newallnodes.reverse()
        rpn = []
        while len(newallnodes) > 0:
            rpn.append(newallnodes[0][0])
            if newallnodes[0][1]:
                self.dfs_for_rpn(
                    input_node_map, output_node_map, newallnodes, rpn, newallnodes[0][0]
                )
            del newallnodes[0]
        rpn.reverse()

        for idx in range(len(rpn)):
            for i in range(idx):
                if isinstance(rpn[i], ir.node.Node) and rpn[i] == rpn[idx]:
                    rpn[idx] = i
                    break

        op_types = []
        for i in range(len(rpn)):
            if isinstance(rpn[i], ir.node.Node):
                op_types.append(self._node_map[rpn[i].__class__])
            elif isinstance(rpn[i], int):
                op_types.append(1000000 + rpn[i])
            else:
                op_types.append(2000000 + inputs_map[rpn[i]])

        nodes.remove(node)
        for n in nodes:
            for name in n.output_names:
                graph.remove_symbol(name)
            graph.remove_node(n)
        index = 0
        for n in graph.nodes:
            if n == node:
                break
            index = index + 1
        for name in node.output_names:
            graph.remove_symbol(name)
        graph.remove_node(node)

        symbols = [graph.get_symbol(name) for name in inputs]
        newinputs = []
        for i, s in zip(inputs, symbols):
            if s.is_value():
                o = graph.context.create_symbol_name(i)
                graph.add_node(
                    index,
                    ir.node.base.ToTensor(
                        node.name + "_FusionOpToTensor_" + str(index), graph, [i], [o]
                    ),
                )
                index = index + 1
                newinputs.append(o)
            else:
                newinputs.append(i)
        inputs = newinputs
        result_shape, broadcast_marks = self.get_result_shape_and_broadcast_marks(
            graph, inputs
        )
        if result_shape:
            graph.add_node(
                index,
                ir.node.fusion.FusionOpFixedShape(
                    node.name + "_FusionOpFixedShape",
                    graph,
                    inputs,
                    outputs,
                    op_types=op_types,
                    result_shape=result_shape,
                    broadcast_marks=broadcast_marks,
                ),
            )
        else:
            graph.add_node(
                index,
                ir.node.fusion.FusionOpDynamicShape(
                    node.name + "_FusionOpDynamicShape",
                    graph,
                    inputs,
                    outputs,
                    op_types=op_types,
                ),
            )
        graph.reflash_symbols()

    def runonce(self, graph: ir.framework.Graph, input_node_map, output_node_map):
        changed = False
        for node in reversed(graph.nodes):
            if node.__class__ in self._node_map:
                allnodes = []
                nodes = set()
                inputs = []
                allnodes.append(node)
                nodes.add(node)
                self.dfs(input_node_map, output_node_map, allnodes, nodes, inputs, node)

                if len(nodes) > 1:
                    changed = True
                    self.gen_new_graph(
                        graph,
                        input_node_map,
                        output_node_map,
                        allnodes,
                        nodes,
                        node,
                        inputs,
                    )
                    return changed
        return changed
