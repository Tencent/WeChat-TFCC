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


class MergeSameNode(Optimizer):
    class MergeNodeWrapper(object):
        def __init__(self, node: ir.node.Node):
            self._node = node

        def key_to_tuple(self, value):
            if isinstance(value, (tuple, list, set)):
                real_value = []
                for v in value:
                    real_value.append(self.key_to_tuple(v))
                return tuple(real_value)
            return value

        def get_key(self):
            attr_list = []
            for name in self._node.attributes:
                value = self._node.attributes[name]
                attr_list.append((name, value))
            key = (
                self._node.__class__,
                tuple(self._node.input_names),
                self.key_to_tuple(attr_list),
                len(self._node.output_names),
            )
            return key

        def __hash__(self):
            return hash(self.get_key())

        def __eq__(self, other):
            return self.get_key() == other.get_key()

    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()
        wrapper_map = {}
        name_map = {}
        for node in graph.nodes:
            if node.attributes is None:
                continue
            wrapper = self.MergeNodeWrapper(node)
            if wrapper not in wrapper_map:
                wrapper_map[wrapper] = node
                continue
            assert len(node.output_names) == len(wrapper_map[wrapper].output_names)
            for src, dst in zip(node.output_names, wrapper_map[wrapper].output_names):
                assert src != dst
                name_map[src] = dst

        # combine rules
        for name in sorted(name_map.keys()):
            target = name_map[name]
            while target in name_map:
                target = name_map[target]
            name_map[name] = target

        # update inputs

        changed = False
        for node in graph.nodes:
            c = False
            inputs = []
            for inp in node.inputs:
                if inp.name in name_map:
                    c = True
                    inputs.append(name_map[inp.name])
                else:
                    inputs.append(inp.name)
            if c:
                node.update_inputs(inputs)
                changed = True

        graph.reflash_symbols()

        if changed:
            logging.debug("{} cost: {}".format(self.__class__, time.time() - ts))
            return True
        else:
            return False

    def __call__(self, model: ir.framework.Model) -> bool:
        changed = False
        for graph in model.graphs.values():
            changed = changed or self.process_graph(graph)
        return changed
