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

import time
import logging
import ir.framework
from ir.optimizer.optimizer import Optimizer


class ChangeIncompleteDataNode(Optimizer):
    def process_node(
        self, shape_name_map: dict, node: ir.node.Node, graph: ir.framework.Graph
    ):
        insert_index = graph.nodes.index(node)
        node_index = 0
        for output in node.outputs:
            graph.remove_symbol(output.name)
            if output.is_value():
                inputs = [shape_name_map[output.incomplete_data[0]].name]
                axis = shape_name_map[output.incomplete_data[0]].shape.index(
                    output.incomplete_data[0]
                )
                get_dimension_node = ir.node.base.GetDimension(
                    node.name + "cidn_get_dimension_" + str(node_index),
                    graph,
                    inputs,
                    [output.name],
                    axis=axis,
                    dtype=output.dtype,
                )
                graph.add_node(insert_index, get_dimension_node)
                insert_index += 1
            elif output.is_vector():
                inputs = []
                vec = []
                for v in output.incomplete_data:
                    if isinstance(v, int):
                        if len(vec) > 0 and isinstance(vec[-1], list):
                            vec[-1].append(v)
                        else:
                            vec.append([v])
                        continue
                    symbol = shape_name_map[v]
                    axis = symbol.shape.index(v)
                    get_dimension_node_output_name = graph.context.create_symbol_name(
                        symbol.name
                    )
                    get_dimension_node = ir.node.base.GetDimension(
                        node.name + "cidn_get_dimension_" + str(node_index),
                        graph,
                        [symbol.name],
                        [get_dimension_node_output_name],
                        axis=axis,
                        dtype=ir.framework.DataType.UINT32,
                    )
                    graph.add_node(insert_index, get_dimension_node)
                    insert_index += 1
                    vec.append(len(inputs))
                    inputs.append(get_dimension_node_output_name)

                create_vector_node = ir.node.base.CreateVector(
                    node.name + "cidn_create_vector",
                    graph,
                    inputs,
                    [output.name],
                    vec=vec,
                    dtype=output.dtype,
                )
                graph.add_node(insert_index, create_vector_node)
                insert_index += 1
            else:
                raise RuntimeError("Unknow error")
        graph.remove_node(node)

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()
        shape_name_map = {}
        for name in graph.inputs:
            symbol = graph.get_symbol(name)
            for s in symbol.shape:
                if isinstance(s, str):
                    shape_name_map[s] = symbol

        node_list = []
        for node in graph.nodes:
            if isinstance(node, (ir.node.base.GetDimension, ir.node.base.CreateVector)):
                continue
            if len(node.output_names) == 0:
                continue
            can_change = True
            for output in node.outputs:
                if not isinstance(output.incomplete_data, list):
                    can_change = False
                    break
                if not output.is_vector() and not output.is_value():
                    can_change = False
                    break
                if not all(
                    [
                        isinstance(v, int) or v in shape_name_map
                        for v in output.incomplete_data
                    ]
                ):
                    can_change = False
                    break
            if not can_change:
                continue
            node_list.append(node)

        for node in node_list:
            self.process_node(shape_name_map, node, graph)

        graph.reflash_symbols()

        if len(node_list) > 0:
            logging.debug(
                "{} process {} cost: {}".format(
                    self.__class__, len(node_list), time.time() - ts
                )
            )
            return True
        else:
            return False

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = changed or self.process_graph(graph)
        return changed
