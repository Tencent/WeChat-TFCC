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
import typing
from ir.framework.symbol import Symbol


class Node(object):
    def __init__(
        self,
        name,
        graph,
        inputs: typing.List[str],
        outputs: typing.List[str],
        *args,
        **kwargs
    ):
        self._name = name
        self._graph = graph
        self._inputs = inputs
        self._outputs = outputs

        self.update_attributes(*args, **kwargs)
        self.update_symbols()

    def create_shape_name(self, prefix):
        return self.graph.context.create_shape_name(prefix)

    def update_attributes(self):
        pass

    def calculatable(self):
        return all(
            [symbol.is_constant() and not symbol.is_complex() for symbol in self.inputs]
        )

    def inference(self):
        raise RuntimeError("Unimplement inference function")

    def calculate(self):
        logging.info("Node: {} may can pre calculate".format(self))

    def calculate_incomplete_data(self):
        pass

    def update_inputs(self, inputs):
        self._inputs = inputs

    def update_outputs(self, outputs):
        self._outputs = outputs

    def update_symbols(self):
        # set output symbols
        for name in self._outputs:
            symbol = Symbol(name)
            self.graph.add_symbol(symbol)

        self.inference()

        if self.calculatable():
            self.calculate()
        else:
            self.calculate_incomplete_data()

        assert all([symbol.verify() for symbol in self.outputs])

    @property
    def name(self) -> str:
        return self._name

    @property
    def graph(self):
        return self._graph

    @property
    def model(self):
        return self.graph.model

    @property
    def inputs(self) -> typing.List[Symbol]:
        symbols = [self.graph.get_symbol(name) for name in self._inputs]
        assert all(symbols)
        return symbols

    @property
    def input_names(self):
        return list(self._inputs)

    @property
    def outputs(self) -> typing.List[Symbol]:
        symbols = [self.graph.get_symbol(name) for name in self._outputs]
        assert all(symbols)
        return symbols

    @property
    def output_names(self):
        return list(self._outputs)

    @property
    def attributes(self):
        raise NotImplementedError

    def __str__(self):
        return (
            "["
            + ", ".join(self._inputs)
            + "] -> "
            + str(self.__class__)
            + " -> ["
            + ", ".join(self._outputs)
            + "]"
        )
