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
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework.ops import Graph, Operation
import ir.framework
from frontend.tensorflow.converter.converter import Converter

from . import tf_utils
from . import common


def add_const_symbol(op: Operation, graph: ir.framework.Graph):
    tensor_pb = op.get_attr("value")
    data = tf_utils.get_tf_tensor_data(tensor_pb)
    shape = []
    for dim in tensor_pb.tensor_shape.dim:
        shape.append(dim.size)
    if not shape:
        shape = [1]
    data = data.reshape(shape)

    symbol = ir.framework.symbol.Symbol(op.outputs[0].name)
    symbol.dtype = common.tf_to_symbol_dtype[tensor_pb.dtype]

    if tensor_pb.tensor_shape.dim:
        symbol.stype = ir.framework.SymbolType.CONSTANT_TENSOR
    elif data.size == 1:
        symbol.stype = ir.framework.SymbolType.CONSTANT_VALUE
    else:
        symbol.stype = ir.framework.SymbolType.CONSTANT_VECTOR
    symbol.shape = shape
    symbol.data = data
    symbol.origin_stype = symbol.stype
    graph.add_symbol(symbol)
    graph.add_keep(symbol.name)


def add_placeholder_symbol(op: Operation, graph: ir.framework.Graph):
    for oup_tensor in op.outputs:
        if oup_tensor.name in graph.symbols:
            continue
        symbol = ir.framework.symbol.Symbol(oup_tensor.name)
        symbol.dtype = common.tf_to_symbol_dtype[oup_tensor.dtype]

        shape = tf_utils.get_tf_tensor_shape(oup_tensor)
        if len(shape) == 0:
            symbol.shape = [1]
            symbol.stype = ir.framework.SymbolType.VALUE
        else:
            symbol.shape = shape
            symbol.stype = ir.framework.SymbolType.VIEW
        graph.add_symbol(symbol)


def add_op_node(
    op: Operation, converters: typing.List[Converter], graph: ir.framework.Graph
):
    succ = False
    for converter in converters:
        if converter(op, graph):
            succ = True
            break
    if not succ:
        logging.debug("Unsupport operation type: {}".format(op.type))
    return succ


def log_all_unsupport_nodes(
    ops: typing.List[Operation], converters: typing.List[Converter]
):
    unsupport_type_set = set()
    for op in ops:
        succ = False
        for converter in converters:
            if converter.may_accept(op):
                succ = True
                break
        if not succ:
            unsupport_type_set.add(op.type)
    logging.debug("May unsupport operation type set: {}".format(unsupport_type_set))
