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


class RemoveUselessReshapeEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.Reshape):
            return None
        if node.inputs[0].shape == node.outputs[0].shape:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessFlattenEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.Flatten):
            return None
        if node.inputs[0].shape == node.outputs[0].shape:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessCastEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.Cast):
            return None
        if node.inputs[0].dtype == node.outputs[0].dtype:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessSliceV2Equivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.SliceV2):
            return None
        if not node.inputs[1].is_constant():
            return None
        if not node.inputs[2].is_constant():
            return None
        start = node.inputs[1].data.tolist()[0]
        end = node.inputs[2].data.tolist()[0]
        if isinstance(node.inputs[0].shape[node.axis], int):
            while start < 0:
                start += node.inputs[0].shape[node.axis]
            while end < 0:
                end += node.inputs[0].shape[node.axis]
        if start != 0:
            return None
        if isinstance(node.inputs[0].shape[node.axis], int):
            if end >= node.inputs[0].shape[node.axis]:
                return {node.output_names[0]: node.input_names[0]}
        if end >= 2147483647:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessReduceEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(
            node,
            (ir.node.math.ReduceMean, ir.node.math.ReduceProd, ir.node.math.ReduceSum),
        ):
            return None
        if node.inputs[0].shape == node.outputs[0].shape:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessIdentityEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.Identity):
            return None
        return {node.output_names[0]: node.input_names[0]}


class RemoveUselessToXEquivalence(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(
            node, (ir.node.base.ToTensor, ir.node.base.ToVector, ir.node.base.ToValue)
        ):
            return None
        if node.inputs[0].stype == node.outputs[0].stype:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessMul(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, (ir.node.math.Mul)):
            return None
        if node.inputs[0].is_constant() and node.inputs[0].data.tolist() == [1]:
            return {node.output_names[0]: node.input_names[1]}
        if node.inputs[1].is_constant() and node.inputs[1].data.tolist() == [1]:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessAdd(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, (ir.node.math.Add)):
            return None
        if node.inputs[0].is_constant() and node.inputs[0].data.tolist() == [0]:
            return {node.output_names[0]: node.input_names[1]}
        if node.inputs[1].is_constant() and node.inputs[1].data.tolist() == [0]:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveUselessTranspose(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(node, ir.node.base.Transpose):
            return None
        axes = []
        for i, p in enumerate(node.perm):
            if i != p:
                axes.append(i)
        shape = node.inputs[0].shape
        for axis in axes:
            if shape[axis] != 1:
                return None
        return {node.output_names[0]: node.input_names[0]}


class RemoveUselessPad(object):
    def __call__(self, node: ir.node.Node) -> dict:
        if not isinstance(
            node,
            (ir.node.math.ReduceMean, ir.node.math.ReduceProd, ir.node.math.ReduceSum),
        ):
            return None
        if node.inputs[0].shape == node.outputs[0].shape:
            return {node.output_names[0]: node.input_names[0]}
        return None


class RemoveNodes(Optimizer):
    def __init__(self):
        self._equivalances = [
            RemoveUselessReshapeEquivalence(),
            RemoveUselessFlattenEquivalence(),
            RemoveUselessCastEquivalence(),
            RemoveUselessSliceV2Equivalence(),
            RemoveUselessReduceEquivalence(),
            RemoveUselessIdentityEquivalence(),
            RemoveUselessToXEquivalence(),
            RemoveUselessMul(),
            RemoveUselessAdd(),
            RemoveUselessTranspose(),
            RemoveUselessPad(),
        ]

    def equivalence_names(self, node: ir.node.Node) -> dict:
        for equivalence in self._equivalances:
            equivalence_map = equivalence(node)
            if equivalence_map:
                return equivalence_map
        return None

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()
        node_list = []
        name_map = {}
        for node in graph.nodes:
            equivalence_map = self.equivalence_names(node)
            if not equivalence_map:
                continue
            assert len(equivalence_map) == len(node.outputs)
            if all([not name in graph.outputs for name in equivalence_map]):
                name_map.update(equivalence_map)
                node_list.append(node)

        # combine rules
        for name in name_map:
            target = name_map[name]
            while target in name_map:
                target = name_map[target]
            name_map[name] = target

        # update inputs
        for node in graph.nodes:
            changed = False
            inputs = []
            for inp in node.inputs:
                if inp.name in name_map:
                    changed = True
                    inputs.append(name_map[inp.name])
                else:
                    inputs.append(inp.name)
            if changed:
                node.update_inputs(inputs)

        for node in node_list:
            graph.remove_node(node)

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


def get_all_optimizers():
    return [
        RemoveNodes(),
    ]
