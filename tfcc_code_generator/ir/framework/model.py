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


class Model(object):
    def __init__(self, context):
        self._context = context
        self._graphs = {}
        self._entrance = None

    def add_graph(self, graph):
        assert graph.name not in self._graphs
        self._graphs[graph.name] = graph

    def verify(self):
        if len(self.graphs) == 0:
            return False
        if not self.context:
            return False
        if not self.entrance:
            return False
        if not all([graph.context == self.context for graph in self.graphs.values()]):
            return False

        for graph in self.graphs.values():
            if not graph.verify():
                return False

        return True

    @property
    def context(self):
        return self._context

    @property
    def graphs(self):
        return self._graphs

    @property
    def entrance(self):
        return self._entrance

    @entrance.setter
    def entrance(self, value):
        self._entrance = value

    @property
    def summary(self):
        return "graph count {} node count {} constant count: {}".format(
            len(self.graphs),
            sum([len(graph.nodes) for graph in self.graphs.values()]),
            sum([len(graph.keep_symbol_names) for graph in self.graphs.values()]),
        )
