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
from ir.node import Node
from .symbol import Symbol
from ir.framework.context import Context
from ir.framework.model import Model


class Graph(object):
    def __init__(self, name: str, model: Model):
        assert isinstance(model, Model)
        self._name = name
        self._nodes = []
        self._symbols = {}
        self._model = model
        self._inputs = []
        self._outputs = []
        self._keep_symbol_names = set()

    def add_keep(self, name):
        self._keep_symbol_names.add(name)

    def add_symbol(self, symbol: Symbol):
        assert symbol.name not in self._symbols
        self._symbols[symbol.name] = symbol

    def get_symbol(self, name: str) -> Symbol:
        if name not in self._symbols:
            return None
        return self._symbols[name]

    def remove_symbol(self, name: str):
        assert name in self._symbols
        self._symbols.pop(name)
        if name in self._keep_symbol_names:
            self._keep_symbol_names.remove(name)

    def reset_symbols(self):
        symbols = {}
        for name in self.inputs:
            symbols[name] = self.symbols[name]
        for name in self._keep_symbol_names:
            symbols[name] = self.symbols[name]
        self._symbols = symbols

    def reflash_symbols(self):
        self.reset_symbols()
        for node in self.nodes:
            node.update_symbols()

    def append_node(self, node: Node):
        self._nodes.append(node)

    def add_node(self, idx: int, node: Node):
        self._nodes.insert(idx, node)

    def remove_node(self, node: Node):
        self._nodes.remove(node)

    def verify(self):
        if not all([name in self._symbols for name in self._inputs]):
            return False
        if not all([name in self._symbols for name in self._outputs]):
            return False
        return True

    @property
    def model(self) -> Model:
        return self._model

    @property
    def context(self) -> Context:
        return self._model.context

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> typing.List[Node]:
        return self._nodes

    @property
    def inputs(self) -> typing.List[str]:
        return self._inputs

    @inputs.setter
    def inputs(self, values):
        assert all([name in self._symbols for name in values])
        self._inputs = values

    @property
    def outputs(self) -> typing.List[str]:
        return self._outputs

    @outputs.setter
    def outputs(self, values):
        assert all([name in self._symbols for name in values])
        self._outputs = values

    @property
    def symbols(self) -> typing.Dict[str, Symbol]:
        return self._symbols

    @property
    def keep_symbol_names(self):
        return self._keep_symbol_names
