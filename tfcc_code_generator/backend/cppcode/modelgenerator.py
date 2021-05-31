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

from .namemanager import NameManager
from ir.framework.model import Model


class ModelGenerator(object):
    def __init__(self, model: Model, entrance_cls_name: str, name_manager: NameManager):
        from .graphgenerator import GraphGenerator

        self._name_manager = name_manager
        self._model = model
        self._graph_generators = {}
        self._graph_generators[model.entrance] = GraphGenerator(
            model.graphs[model.entrance], entrance_cls_name, self
        )
        for graph_name in model.graphs:
            if graph_name == model.entrance:
                continue
            self._graph_generators[graph_name] = GraphGenerator(
                model.graphs[graph_name], model.graphs[graph_name].name, self
            )

    @property
    def name_manager(self):
        return self._name_manager

    @property
    def model(self):
        return self._model

    @property
    def graph_generators(self):
        return self._graph_generators

    @property
    def entrance_graph_generator(self):
        return self._graph_generators[self.model.entrance]

    @property
    def declaration(self):
        code = self.entrance_graph_generator.declaration
        for name in self.graph_generators:
            if name == self.model.entrance:
                continue
            code += "\n" + self.graph_generators[name].declaration
        return code

    @property
    def define(self):
        code = self.entrance_graph_generator.define
        for name in self.graph_generators:
            if name == self.model.entrance:
                continue
            code += "\n" + self.graph_generators[name].define
        return code

    @property
    def data(self):
        data = {}
        for graph_generator in self.graph_generators.values():
            data.update(graph_generator.data)
        return data
