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


class RemoveFreeConstant(Optimizer):
    def process_graph(self, graph: ir.framework.Graph):
        ts = time.time()
        name_set = set(graph.outputs)
        for node in graph.nodes:
            name_set.update(node.input_names)

        symbol_name_list = []

        for symbol in graph.symbols.values():
            if not symbol.is_constant():
                continue
            if symbol.name not in graph.keep_symbol_names:
                continue
            if symbol.name in name_set:
                continue
            symbol_name_list.append(symbol.name)

        for name in symbol_name_list:
            graph.remove_symbol(name)

        graph.reflash_symbols()

        if len(symbol_name_list) > 0:
            logging.debug(
                "{} process {} cost: {}".format(
                    self.__class__, len(symbol_name_list), time.time() - ts
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
