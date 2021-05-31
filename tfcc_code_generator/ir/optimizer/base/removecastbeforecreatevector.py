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
import ir.node
from ir.optimizer.optimizer import Optimizer


class RemoveCastBeforeCreateVector(Optimizer):
    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()
        cast_name_map = {}
        change_count = 0
        for node in graph.nodes:
            if isinstance(node, ir.node.base.Cast):
                cast_name_map[node.output_names[0]] = node.input_names[0]
            if isinstance(node, ir.node.base.CreateVector):
                input_names = []
                for name in node.input_names:
                    if name in cast_name_map:
                        input_names.append(cast_name_map[name])
                    else:
                        input_names.append(name)
                if input_names != node.input_names:
                    change_count += 1
                    node.update_inputs(input_names)

        if change_count > 9:
            graph.reflash_symbols()
            logging.debug(
                "{} process {} cost: {}".format(
                    self.__class__, change_count, time.time() - ts
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
