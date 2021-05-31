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

import enum
import time
import logging
import ir.framework
import ir.node
from ir.optimizer.optimizer import Optimizer


class RemoveUnUsedLoopOutputs(Optimizer):
    def process_loop(self, node: ir.node.base.Loop, name_set: set) -> dict:
        if node.scan_count <= 0:
            return False

        remove_output_indices = set()
        for i, name in enumerate(node.output_names[-node.scan_count :]):
            if name not in name_set:
                remove_output_indices.add(i)
        if len(remove_output_indices) == 0:
            return False

        # remove output from node
        new_node_outputs = node.output_names[: node.carried_count]
        for i, name in enumerate(node.output_names[node.carried_count :]):
            if i not in remove_output_indices:
                new_node_outputs.append(name)
        node.update_outputs(new_node_outputs)

        # remove output from graph
        new_graph_outputs = node.model.graphs[node.sub_graph_name].outputs[
            : 1 + node.carried_count
        ]
        for i, name in enumerate(
            node.model.graphs[node.sub_graph_name].outputs[1 + node.carried_count :]
        ):
            if i not in remove_output_indices:
                new_graph_outputs.append(name)
        node.model.graphs[node.sub_graph_name].outputs = new_graph_outputs

        return True

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()

        name_set = set(graph.outputs)
        for node in graph.nodes:
            name_set.update(node.input_names)

        changed = False
        for node in graph.nodes:
            if isinstance(node, ir.node.base.Loop):
                changed = self.process_loop(node, name_set)
                if changed:
                    break

        graph.reflash_symbols()

        if changed:
            logging.debug(
                "{} process {} cost: {}".format(self.__class__, 1, time.time() - ts)
            )
            return True
        else:
            return False

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = self.process_graph(graph)
            if changed:
                break
        return changed


class RemoveUnUsedLoopCaptures(Optimizer):
    def process_loop(self, node: ir.node.base.Loop) -> dict:
        if node.capture_count <= 0:
            return False

        name_set = set(node.model.graphs[node.sub_graph_name].outputs)
        for x in node.model.graphs[node.sub_graph_name].nodes:
            name_set.update(x.input_names)

        remove_output_indices = set()
        for i, name in enumerate(
            node.input_names[1 + node.carried_count :][: node.capture_count]
        ):
            if name not in name_set:
                remove_output_indices.add(i)
        if len(remove_output_indices) == 0:
            return False

        # remove capture from node
        new_node_inputs = []
        for i, name in enumerate(node.input_names):
            if i + 1 + node.carried_count not in remove_output_indices:
                new_node_inputs.append(name)
        node.update_inputs(new_node_inputs)

        # remove output from graph
        new_graph_inputs = []
        for i, name in enumerate(node.model.graphs[node.sub_graph_name].inputs):
            if i + 2 + node.carried_count not in remove_output_indices:
                new_graph_inputs.append(name)
        node.model.graphs[node.sub_graph_name].inputs = new_graph_inputs

        return True

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()

        name_set = set(graph.outputs)
        for node in graph.nodes:
            name_set.update(node.input_names)

        changed = False
        for node in graph.nodes:
            if isinstance(node, ir.node.base.Loop):
                changed = self.process_loop(node, name_set)
                if changed:
                    break

        graph.reflash_symbols()

        if changed:
            logging.debug(
                "{} process {} cost: {}".format(self.__class__, 1, time.time() - ts)
            )
            return True
        else:
            return False

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = self.process_graph(graph)
            if changed:
                break
        return changed


class MoveNodeFromLoopBodyToParent(Optimizer):
    def process_loop(self, node: ir.node.base.Loop) -> dict:
        if node.capture_count <= 0:
            return False

        sub_graph = node.model.graphs[node.sub_graph_name]

        captures = set(sub_graph.inputs[-node.capture_count :])
        movable_node = None
        for x in sub_graph.nodes:
            if isinstance(x, ir.node.base.Identity):
                continue
            if all([name in captures for name in x.input_names]):
                movable_node = x
                break
        if not movable_node:
            return False

        new_node_inputs = []
        for name in movable_node.input_names:
            index = sub_graph.inputs.index(name) - node.carried_count - 2
            new_node_inputs.append(node.input_names[index + 1 + node.carried_count])

        new_node_outputs = [
            node.graph.context.create_symbol_name(name)
            for name in movable_node.output_names
        ]
        new_node = movable_node.__class__(
            movable_node.name + "_move",
            node.graph,
            new_node_inputs,
            new_node_outputs,
            **movable_node.attributes
        )

        # add node
        node.graph.add_node(node.graph.nodes.index(node), new_node)

        # set new capture
        node.update_inputs(
            node.input_names[: 1 + node.carried_count]
            + new_node_outputs
            + node.input_names[1 + node.carried_count :]
        )
        node.update_attributes(
            node.sub_graph_name,
            node.carried_count,
            node.capture_count + len(new_node_outputs),
            node.scan_count,
        )
        sub_graph_new_inputs = []
        sub_graph_new_inputs += sub_graph.inputs[: 2 + node.carried_count]
        sub_graph_new_inputs += [
            sub_graph.context.create_symbol_name(name) for name in new_node_outputs
        ]
        sub_graph_new_inputs += sub_graph.inputs[2 + node.carried_count :]
        sub_graph.inputs = sub_graph_new_inputs

        return True

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()

        name_set = set(graph.outputs)
        for node in graph.nodes:
            name_set.update(node.input_names)

        changed = False
        for node in graph.nodes:
            if isinstance(node, ir.node.base.Loop):
                changed = self.process_loop(node, name_set)
                if changed:
                    break

        graph.reflash_symbols()

        if changed:
            logging.debug(
                "{} process {} cost: {}".format(self.__class__, 1, time.time() - ts)
            )
            return True
        else:
            return False

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = self.process_graph(graph)
            if changed:
                break
        return changed


def get_all_optimizers():
    return [
        RemoveUnUsedLoopOutputs(),
        # RemoveUnUsedLoopCaptures(),
    ]
