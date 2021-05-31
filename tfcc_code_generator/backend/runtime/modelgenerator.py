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

import numpy as np
from backend.runtime.graphgenerator import NameManager
from backend.runtime.graphgenerator import GraphGenerator
from ir.framework.model import Model
from .proto import model_pb2


class ModelGenerator(object):
    def __init__(self, model: Model, name_manager: NameManager):
        self._model = model
        self._name_manager = name_manager

    def process(self):
        model_pb = model_pb2.Model()
        model_pb.proto_version = 1
        data_map = {}
        for graph in self._model.graphs.values():
            graph_generator = GraphGenerator(graph, self._name_manager)
            graph_pb, graph_data = graph_generator.process()
            for name in graph_data:
                if graph_data[name].dtype == np.bool:
                    data_map[graph_pb.name + "/" + name] = graph_data[name].astype(
                        np.uint8
                    )
                else:
                    data_map[graph_pb.name + "/" + name] = graph_data[name]
            model_pb.graphs.append(graph_pb)
            if graph.name == self._model.entrance:
                model_pb.entrance_graph_name = graph_pb.name

        return model_pb, data_map
