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


class MergeSameConstant(Optimizer):
    class MergeSymbolWrapper:
        def __init__(self, symbol: ir.framework.Symbol):
            self._symbol = symbol

        def get_key(self):
            assert self._symbol.is_constant()
            key = (
                self._symbol.dtype,
                self._symbol.stype,
                tuple(self._symbol.shape),
                tuple(self._symbol.data.tobytes()),
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
        for symbol in sorted(graph.symbols.values(), key=lambda a: a.name):
            if not symbol.is_constant():
                continue
            if symbol.name not in graph.keep_symbol_names:
                continue
            if symbol.data.size > 1024:
                continue
            wrapper = self.MergeSymbolWrapper(symbol)
            if wrapper not in wrapper_map:
                wrapper_map[wrapper] = symbol
                continue
            name_map[symbol.name] = wrapper_map[wrapper].name

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
