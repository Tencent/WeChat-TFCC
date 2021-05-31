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

import sys
import time
import typing
import logging
import argparse
import onnx
import onnx.shape_inference
import onnx.version_converter
import ir.framework
from .converter import get_all_converters
from .common import onnx_tensor_to_symbol, onnx_value_info_to_symbol
from ir.framework.context import Context
from ir.framework.model import Model


def logging_all_unsupport_name(nodes, converters):
    unsupport_set = set()
    for node in nodes:
        succ = False
        for converter in converters:
            succ = converter.accept(node)
            if succ:
                break
        if not succ:
            key = (node.domain, node.op_type)
            if key not in unsupport_set:
                unsupport_set.add(key)
                logging.debug(
                    "Unsupport operation type: {}/{}".format(node.domain, node.op_type)
                )


def sort_nodes(graph_proto, captures: set):
    name_set = set(captures)
    name_set.update([initializer.name for initializer in graph_proto.initializer])
    name_set.update([inp.name for inp in graph_proto.input])
    old_nodes = list(graph_proto.node)
    nodes = []
    succ = True
    while succ:
        succ = False
        for node in old_nodes:
            if all([inp in name_set for inp in node.input]):
                nodes.append(node)
                old_nodes.remove(node)
                name_set.update(node.output)
                succ = True
                break
    if len(old_nodes) > 0:
        raise RuntimeError("Reorder nodes error.")
    del graph_proto.node[:]
    graph_proto.node.extend(nodes)
    return graph_proto


def graph2ir(
    name: str,
    graph_proto: onnx.GraphProto,
    model: Model,
    op_set: dict,
    parent: ir.framework.Graph = None,
):
    graph = ir.framework.Graph(name, model)
    for initializer in graph_proto.initializer:
        symbol = onnx_tensor_to_symbol(initializer)
        graph.add_symbol(symbol)
        graph.add_keep(symbol.name)
    inputs = []
    for inp in graph_proto.input:
        if inp.name in graph.symbols:
            continue
        symbol = onnx_value_info_to_symbol(inp)
        if symbol.shape == []:
            symbol.shape = [1]
            symbol.stype = ir.framework.SymbolType.VALUE
        else:
            symbol.stype = ir.framework.SymbolType.VIEW
        graph.add_symbol(symbol)
        inputs.append(symbol.name)

    if "name_map" not in model.context.frontend:
        frontend_data = model.context.frontend
        frontend_data["name_map"] = []
        model.context.frontend = frontend_data

    # set capture
    captures = set()
    name_map = {}
    if parent:
        in_graph_names = set([inp.name for inp in graph_proto.input])
        used_names = set([out.name for out in graph_proto.output])
        for node in graph_proto.node:
            in_graph_names.update(node.output)
            used_names.update(node.input)
        captures = used_names - in_graph_names
        for name in captures:
            parent_symbol = parent.get_symbol(name)
            new_name = model.context.create_symbol_name(name)
            symbol = ir.framework.Symbol(new_name)
            symbol.shape = parent_symbol.shape
            symbol.dtype = parent_symbol.dtype
            if parent_symbol.is_tensor():
                symbol.stype = ir.framework.SymbolType.VIEW
            elif parent_symbol.is_value():
                symbol.stype = ir.framework.SymbolType.VALUE
            elif parent_symbol.is_vector():
                symbol.stype = ir.framework.SymbolType.VECTOR
            else:
                raise RuntimeError("Unknow error.")
            graph.add_symbol(symbol)
            inputs.append(new_name)
            name_map[name] = new_name
    model.context.frontend["name_map"].append(name_map)

    real_name_map = {}
    for v in model.context.frontend["name_map"]:
        real_name_map.update(v)
    for name in real_name_map:
        target = real_name_map[name]
        while target in real_name_map:
            target = real_name_map[name]
        real_name_map[name] = target

    converters = get_all_converters(op_set)
    for i, node_proto in enumerate(graph_proto.node):
        succ = False
        node_proto.input[:] = [
            real_name_map[name] if name in real_name_map else name
            for name in node_proto.input
        ]
        for converter in converters:
            succ = converter(node_proto, graph_proto, graph)
            if succ:
                break
        if not succ:
            logging_all_unsupport_name(graph_proto.node, converters)
            raise RuntimeError(
                "Unknow node:\n{}\nOperation set:\n{}".format(
                    node_proto, op_set[node_proto.domain]
                )
            )

    graph.inputs = inputs
    graph.outputs = [symbol.name for symbol in graph_proto.output]
    if not parent:
        graph.outputs = list(set(graph.outputs))

    model.add_graph(graph)

    model.context.frontend["name_map"].pop()

    return graph, name_map


def onnx2ir(model_proto: onnx.ModelProto) -> Model:
    context = Context()
    model = Model(context)

    op_set = {x.domain: x for x in model_proto.opset_import}
    model.context.frontend_name = "onnx"
    model.context.frontend = {"op_set": op_set}

    graph, _ = graph2ir(
        context.create_graph_name(model_proto.graph.name),
        model_proto.graph,
        model,
        op_set,
    )
    model.entrance = graph.name
    return model


def entrance(args):
    model = onnx.load(args.onnx_path)
    return onnx2ir(model)


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--onnx-path", required=True, help="The path of onnx model")
    return parser
