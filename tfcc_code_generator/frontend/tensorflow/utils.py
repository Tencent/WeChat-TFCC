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
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework.ops import Graph, Operation
from distutils.version import LooseVersion


def node_name(name):
    """Get node name without io#."""
    pos = name.find(":")
    if pos >= 0:
        return name[:pos]
    return name


def split_nodename_and_shape(name):
    """input name with shape into name and shape."""
    # pattern for a node name
    inputs = []
    shapes = {}
    # input takes in most cases the format name:0, where 0 is the output number
    # in some cases placeholders don't have a rank which we can't handle so we let uses override the shape
    # by appending the same, ie: [1, 28, 28, 3]
    name_pattern = r"(?:([\w\d/\-\._:]+)(\[[\-\d,]+\])?),?"
    splits = re.split(name_pattern, name)
    for i in range(1, len(splits), 3):
        inputs.append(splits[i])
        if splits[i + 1] is not None:
            shapes[splits[i]] = [int(n) for n in splits[i + 1][1:-1].split(",")]
    if not shapes:
        shapes = None
    return inputs, shapes


def print_graph_node(graph_node: NodeDef):
    logging.debug("name: {}".format(graph_node.name))
    logging.debug("op: {}".format(graph_node.op))
    for i, inp in enumerate(graph_node.input):
        logging.debug("input {}: {}".format(i, inp))
    for i, (k, v) in enumerate(graph_node.attr.items()):
        logging.debug("Attr {}".format(i))
        logging.debug("key: {}".format(k))
        if k == "value" and v.HasField("tensor"):
            # Do not print tensor_content
            logging.debug("tensor.dtype", v.tensor.dtype)
            logging.debug("tensor.tensor_shape", v.tensor.tensor_shape)
            logging.debug("tensor.tensor_content is ignored")
        else:
            logging.debug(v)
    logging.debug("-------print_graph_node done--------")


def print_graph_def(graph_def: GraphDef):
    for n in graph_def.node:
        print_graph_node(n)


def try_to_print_attr(op: Operation, attr):
    try:
        logging.debug(
            "{} {} {}".format(attr, op.get_attr(attr), type(op.get_attr(attr)))
        )
    except Exception as e:
        logging.debug(e)


def print_tf_op(op: Operation):
    logging.debug("op.name {}".format(op.name))
    logging.debug("op.inputs {}".format(op.inputs))
    if LooseVersion(tf.__version__) < LooseVersion("1.13"):
        if isinstance(op.inputs, Operation._InputList):
            logging.debug("op.inputs.inputs {}".format(op.inputs._inputs))
    logging.debug("op.outputs {}".format(op.outputs))
    logging.debug("op.type {}".format(op.type))  # op type

    # See https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/conv2-d to view attributes of each op
    try_to_print_attr(op, "dtype")
    try_to_print_attr(op, "shape")
    try_to_print_attr(op, "data_format")

    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
    # dtype, tensor_shape, tensor_content
    try:
        logging.debug("value dtype {}".format(op.get_attr("value").dtype))
        logging.debug("value tensor_shape {}".format(op.get_attr("value").tensor_shape))
        logging.debug("value tensor_content is ignored")
    except Exception as e:
        logging.debug(e)
    logging.debug("----------------")


def print_tf_graph(graph: Graph):
    for op in graph.get_operations():
        print_tf_op(op)
