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
import ir.framework
import ir.common
from .proto import model_pb2, common_pb2
from .common import data_type_to_proto
from .namemanager import NameManager
from .nodegenerator import get_all_node_generators


class GraphGenerator(object):
    def __init__(self, graph: ir.framework.Graph, name_manager: NameManager):
        self._graph = graph
        self._name_manager = name_manager

    def transform_symbol(self, name: str) -> model_pb2.Symbol:
        symbol = self._graph.get_symbol(name)
        symbol_proto = model_pb2.Symbol()
        symbol_proto.name = symbol.name
        symbol_proto.data_type = data_type_to_proto(symbol.dtype)
        for s in symbol.shape:
            dimension = model_pb2.Shape.Dimension()
            if isinstance(s, int):
                dimension.value = s
            elif isinstance(s, str):
                dimension.param = s
            else:
                raise RuntimeError("Unknow error")
            symbol_proto.shape.dimensions.append(dimension)

        ref_name = None
        data = None
        stype = symbol.origin_stype
        if not stype:
            stype = symbol.stype

        if stype == ir.framework.SymbolType.VARIABLE:
            symbol_proto.variable.SetInParent()
        elif stype == ir.framework.SymbolType.VIEW:
            symbol_proto.view.SetInParent()
        elif stype == ir.framework.SymbolType.CONSTANT_TENSOR:
            assert symbol.stype == symbol.origin_stype
            assert symbol.name in self._graph.keep_symbol_names
            symbol_proto.constant_tensor.SetInParent()
            symbol_proto.constant_tensor.ref = self._name_manager.get_symbol_name(
                symbol
            )
            ref_name = self._name_manager.get_symbol_name(symbol)
            data = symbol.data
        elif stype == ir.framework.SymbolType.VALUE:
            symbol_proto.value.SetInParent()
        elif stype == ir.framework.SymbolType.CONSTANT_VALUE:
            assert symbol.name in self._graph.keep_symbol_names
            symbol_proto.constant_value.SetInParent()
            symbol_proto.constant_value.ref = self._name_manager.get_symbol_name(symbol)
            ref_name = self._name_manager.get_symbol_name(symbol)
            data = symbol.data
        elif stype == ir.framework.SymbolType.VECTOR:
            symbol_proto.vector.SetInParent()
        elif stype == ir.framework.SymbolType.CONSTANT_VECTOR:
            assert symbol.name in self._graph.keep_symbol_names
            symbol_proto.constant_vector.SetInParent()
            symbol_proto.constant_vector.ref = self._name_manager.get_symbol_name(
                symbol
            )
            ref_name = self._name_manager.get_symbol_name(symbol)
            data = symbol.data
        else:
            raise RuntimeError("Unknow symbol type {}".format(stype))

        return symbol_proto, ref_name, data

    def process(self):
        graph_proto = model_pb2.Graph()
        graph_proto.name = self._graph.name
        graph_proto.inputs[:] = self._graph.inputs
        graph_proto.outputs[:] = self._graph.outputs

        name_set = set()
        name_set.update(self._graph.symbols.keys())

        data_map = {}

        for name in name_set:
            symbol_proto, ref_name, data = self.transform_symbol(name)
            graph_proto.symbols.append(symbol_proto)
            if ref_name is not None:
                data_map[ref_name] = data

        generator_classes = get_all_node_generators()

        all_succ = True
        for node in self._graph.nodes:
            generator = None
            for generator_class in generator_classes:
                if generator_class.accept(node):
                    generator = generator_class(node)
                    break
            if generator:
                node_proto = generator.proto
                graph_proto.nodes.append(node_proto)
            else:
                all_succ = False
                logging.debug(node.__class__)
        if not all_succ:
            raise RuntimeError("Node to proto error")

        return graph_proto, data_map
