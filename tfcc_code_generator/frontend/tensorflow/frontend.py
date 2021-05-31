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

import argparse

import tensorflow as tf
from tensorflow.python.framework.ops import Graph, Operation

import ir.framework
from ir.framework.context import Context
from ir.framework.model import Model

from . import tf_loader
from .tf_parser import (
    add_const_symbol,
    add_placeholder_symbol,
    add_op_node,
    log_all_unsupport_nodes,
)
from .converter import get_all_converters
from . import utils


def graph2ir(tf_graph: Graph, name, inputs, outputs):
    context = Context()
    model = Model(context)
    model.context.frontend_name = "tensorflow"

    ir_graph = ir.framework.Graph(context.create_graph_name(name), model)
    converters = get_all_converters()
    ops = tf_graph.get_operations()

    # Add Const symbol first
    inputs = []
    for op in ops:
        if op.type == "Const":
            add_const_symbol(op, ir_graph)
        elif op.type == "Placeholder":
            add_placeholder_symbol(op, ir_graph)
            inputs.append(op.name + ":0")

    ops = [op for op in ops if op.type not in ["Const", "Placeholder", "NoOp"]]

    for op in ops:
        succ = add_op_node(op, converters, ir_graph)
        if not succ:
            log_all_unsupport_nodes(ops, converters)
            raise RuntimeError("Unknow node: {}".format(op))

    ir_graph.inputs = inputs
    ir_graph.outputs = outputs

    model.add_graph(ir_graph)
    model.entrance = ir_graph.name
    return model


def change_symbol_name(graph: ir.framework.Graph, src_name: str, dst_name: str):
    if src_name == dst_name:
        return
    assert dst_name not in graph.symbols
    if src_name in graph.inputs:
        src_symbol = graph.get_symbol(src_name)
        dst_symbol = ir.framework.Symbol(dst_name)
        src_symbol.copy_to(dst_symbol)
        graph.inputs[graph.inputs.index(src_name)] = dst_name
        graph.remove_symbol(src_name)
        graph.add_symbol(dst_symbol)

    if src_name in graph.keep_symbol_names:
        src_symbol = graph.get_symbol(src_name)
        dst_symbol = ir.framework.Symbol(dst_name)
        src_symbol.copy_to(dst_symbol)
        graph.remove_symbol(src_name)
        graph.add_symbol(dst_symbol)
        graph.add_keep(dst_symbol.name)

    for node in graph.nodes:
        if src_name in node.input_names:
            new_input_names = [
                dst_name if name == src_name else name for name in node.input_names
            ]
            node.update_inputs(new_input_names)
        if src_name in node.output_names:
            new_output_names = [
                dst_name if name == src_name else name for name in node.output_names
            ]
            node.update_outputs(new_output_names)
    if src_name in graph.outputs:
        graph.outputs[graph.outputs.index(src_name)] = dst_name
    graph.reflash_symbols()


def entrance(args):
    # TODO
    if args.inputs:
        newinputs = []
        for inp in args.inputs:
            newinputs.append(inp if ":" in inp else inp + ":0")
        args.inputs = newinputs if newinputs else None
    if args.outputs:
        newoutputs = []
        for out in args.outputs:
            if out.strip() != "":
                newoutputs.append(out if ":" in out else out + ":0")
        args.outputs = newoutputs if newoutputs else None

    # TODO: tf.get_logger() not available in tf1
    # tf.get_logger().setLevel('ERROR')
    name_map = {}
    if args.tf_model_type == "graphdef":
        graphdef, inputs, outputs = tf_loader.from_graphdef(
            args.tf_model_path, args.inputs, args.outputs
        )
    elif args.tf_model_type == "checkpoint":
        graphdef, inputs, outputs = tf_loader.from_checkpoint(
            args.tf_model_path, args.inputs, args.outputs
        )
    elif args.tf_model_type == "saved_model":
        graphdef, inputs, outputs, name_map = tf_loader.from_saved_model(
            args.tf_model_path, args.inputs, args.outputs, args.tag, args.signature_def
        )
    elif args.tf_model_type == "keras":
        graphdef, inputs, outputs = tf_loader.from_keras(
            args.tf_model_path, args.inputs, args.outputs, args.keep_keras_weights_name
        )
    else:
        raise NotImplementedError("invalid tf model type")

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graphdef, name="")
        # if args.debug:
        #     utils.print_tf_graph(graph)
        model = graph2ir(graph, "TFModel", inputs, outputs)

    if args.symbol_name_type == "signature":
        for src_name, dst_name in name_map.items():
            entrance_graph = model.graphs[model.entrance]
            assert isinstance(entrance_graph, ir.framework.Graph)
            if dst_name in entrance_graph.symbols:
                change_symbol_name(
                    entrance_graph, dst_name, model.context.create_symbol_name(dst_name)
                )
            change_symbol_name(entrance_graph, src_name, dst_name)
    return model


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--tf-model-path", required=True, help="The path of tensorflow model"
    )
    parser.add_argument(
        "--tf-model-type",
        required=True,
        choices=["graphdef", "checkpoint", "saved_model", "keras"],
        help="Graph type",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=False,
        help="List of input names, seperated by space",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=False,
        help="List of output names, seperated by space",
    )
    parser.add_argument("--tag", required=False, help='"tag" to use for saved_model')
    parser.add_argument(
        "--signature_def",
        required=False,
        help='"signature_def" from saved_model to use',
    )
    parser.add_argument(
        "--symbol-name-type",
        default="tensor",
        choices=["tensor", "signature"],
        help='Input/Output name type. "tensor" use name of tensor, "signature" use signature key',
    )
    parser.add_argument(
        "--keep-keras-weights-name",
        action="store_true",
        help="Whether keep keras weights name",
    )

    return parser
