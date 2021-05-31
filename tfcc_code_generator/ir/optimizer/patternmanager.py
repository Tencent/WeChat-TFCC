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
import ir.optimizer.pattern
import ir.optimizer.patterngroup
from ir.optimizer.nodemanager import NodeManager


class PatternManager(Optimizer):
    def __init__(self, group: ir.optimizer.patterngroup.PatternGroup):
        super().__init__()
        self._patterns = [
            pattern
            for pattern in ir.optimizer.pattern.get_all_patterns()
            if pattern.pattern_group == group
        ]

    def process_graph(self, graph: ir.framework.Graph):
        node_manager = NodeManager(graph)
        changed = False
        nodes = set()
        for pattern in self._patterns:
            may_useless_nodes = pattern.process(node_manager)
            if may_useless_nodes is None:
                continue
            changed = True
            nodes.update(may_useless_nodes)

        if nodes:
            ts = time.time()
            remove_count = self.remove_useless_nodes(node_manager, nodes)
            logging.debug(
                "PatternManager remove node count: {} cost: {:.4} s".format(
                    remove_count, time.time() - ts
                )
            )

        return changed

    def remove_useless_nodes(self, node_manager: NodeManager, may_useless_nodes: set):
        remove_count = 0
        changed = True
        while changed:
            changed = False
            inputs = set(node_manager.graph.outputs)
            for node in node_manager.graph.nodes:
                inputs.update(node.input_names)
            useless_nodes = set()
            for node in may_useless_nodes:
                if node not in node_manager.graph.nodes:
                    continue
                if all([not name in inputs for name in node.output_names]):
                    useless_nodes.add(node)
            if not useless_nodes:
                continue
            changed = True
            for node in useless_nodes:
                node_manager.graph.remove_node(node)
            remove_count += len(useless_nodes)
        return remove_count

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = changed or self.process_graph(graph)
        return changed


def get_all_pattern_managers():
    managers = []
    for group in ir.optimizer.patterngroup.PatternGroup:
        managers.append(PatternManager(group))
    return managers
