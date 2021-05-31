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
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
from tensorflow.python.framework.ops import control_dependencies

from .tf_utils import get_tf_version, get_tf_nodedef_by_name, tensor_proto_to_ndarray
from . import utils


def is_tf2():
    return tf.__version__.startswith("2.")


def _not_implemented_tf_placeholder(name):
    """Creates a placeholder function for missing Tensorflow imports"""

    def not_implemented_tf_placeholder(*args, **kwargs):
        raise NotImplementedError(
            f"Tensorflow verison {tf.__version__} does not implement "
            f"`{name}`, try converting your model with a different version."
        )

    return not_implemented_tf_placeholder


if is_tf2():
    convert_variables_to_constants = (
        tf.compat.v1.graph_util.convert_variables_to_constants
    )
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )
else:
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    convert_variables_to_constants_v2_as_graph = _not_implemented_tf_placeholder(
        "convert_variables_to_constants_v2_as_graph"
    )
    convert_variables_to_constants_v2 = _not_implemented_tf_placeholder(
        "convert_variables_to_constants_v2"
    )

if is_tf2():
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session
    tf_graphdef = tf.compat.v1.GraphDef
    tf_nodedef = tf.compat.v1.NodeDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.io.gfile
    tf_placeholder = tf.compat.v1.placeholder
    tf_extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
    tf_attr_value = tf.compat.v1.AttrValue
elif LooseVersion(tf.__version__) >= "1.13":
    # 1.13 introduced the compat namespace
    tf_reset_default_graph = tf.compat.v1.reset_default_graph
    tf_global_variables = tf.compat.v1.global_variables
    tf_session = tf.compat.v1.Session
    tf_graphdef = tf.compat.v1.GraphDef
    tf_nodedef = tf.compat.v1.NodeDef
    tf_import_meta_graph = tf.compat.v1.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.compat.v1.placeholder
    tf_extract_sub_graph = tf.compat.v1.graph_util.extract_sub_graph
    tf_attr_value = tf.compat.v1.AttrValue
else:
    # older than 1.13
    tf_reset_default_graph = tf.reset_default_graph
    tf_global_variables = tf.global_variables
    tf_session = tf.Session
    tf_graphdef = tf.GraphDef
    tf_nodedef = tf.NodeDef
    tf_import_meta_graph = tf.train.import_meta_graph
    tf_gfile = tf.gfile
    tf_placeholder = tf.placeholder
    tf_extract_sub_graph = tf.graph_util.extract_sub_graph
    tf_attr_value = tf.AttrValue


def inputs_without_resource(sess, input_names):
    try:
        new_input_names = []
        for n in input_names:
            t = sess.graph.get_tensor_by_name(n)
            if t.dtype != tf.dtypes.resource:
                new_input_names.append(n)
        input_names = new_input_names
    except:
        pass
    return input_names


def from_function(func, input_names, output_names):
    if get_tf_version() < LooseVersion("2.2"):
        frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
    else:
        frozen_func = convert_variables_to_constants_v2(
            func, lower_control_flow=False, aggressive_inlining=True
        )
    graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    tf_reset_default_graph()
    with tf_session() as sess:
        tf.import_graph_def(graph_def, name="")
        input_names = inputs_without_resource(sess, input_names)
    return graph_def


def freeze_session(sess, input_names=None, output_names=None):
    """Freezes the state of a session into a pruned computation graph."""
    output_node_names = [i.split(":")[:-1][0] for i in output_names]
    keep_var_names = [i.split(":")[:-1][0] for i in input_names]
    with sess.graph.as_default():
        output_node_names = output_node_names or []
        output_node_names += [v.op.name for v in tf_global_variables()]
        output_node_names += keep_var_names
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        graph_def = convert_variables_to_constants(sess, graph_def, output_node_names)
    return graph_def


def remove_redundant_nodes(frozen_graph: GraphDef, input_names, output_names):
    output_node_names = [i.split(":")[:-1][0] for i in output_names]
    keep_var_names = [i.split(":")[:-1][0] for i in input_names]
    queue = output_node_names[:]
    reserved_node_names = []
    reserved_nodes = []
    while len(queue) > 0:
        op_name = queue.pop(0)
        if op_name in reserved_node_names:
            continue
        node = get_tf_nodedef_by_name(frozen_graph, op_name)
        if op_name in keep_var_names and node.op != "Placeholder":
            # convert the input node to a placeholder
            origin_shape = TensorShapeProto()
            if len(node.attr["_output_shapes"].list.shape) > 0:
                origin_shape = node.attr["_output_shapes"].list.shape[0]
            elif node.op == "Const" and node.attr["value"]:
                origin_shape = node.attr["value"].tensor.tensor_shape
            dtype = node.attr["T"].type
            if node.op == "Const":
                dtype = node.attr["dtype"].type
            placeholder_node = tf_nodedef(
                name=op_name,
                op="Placeholder",
                attr={
                    "dtype": tf_attr_value(type=dtype),
                    "shape": tf_attr_value(shape=origin_shape),
                },
            )
            reserved_node_names.append(op_name)
            reserved_nodes.append(placeholder_node)
        else:
            reserved_nodes.append(node)
            for inp in node.input:
                if inp[0] == "^":
                    inp = inp[1:]
                queue.append(inp.split(":")[0])
                if "ParseExample" in queue[-1]:
                    logging.warn(
                        "ParseExample in reduced graph, name {}".format(queue[-1])
                    )
            reserved_node_names.append(op_name)
    reserved_nodes.reverse()
    reserved_graph_def = tf_graphdef()
    reserved_graph_def.node.extend(reserved_nodes)

    return reserved_graph_def


def remove_redundant_inputs(frozen_graph, input_names):
    """Remove redundant inputs not in frozen graph."""
    frozen_inputs = []
    # get inputs in frozen graph
    for n in frozen_graph.node:
        for inp in input_names:
            if utils.node_name(inp) == n.name:
                frozen_inputs.append(inp)
    deleted_inputs = list(set(input_names) - set(frozen_inputs))
    if deleted_inputs:
        raise RuntimeError("inputs {} is not in frozen graph".format(deleted_inputs))
    return frozen_inputs


def from_graphdef(model_path, input_names, output_names):
    """Load tensorflow graph from graphdef."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    with tf_session() as sess:
        graph_def = tf_graphdef()
        with tf_gfile.GFile(model_path, "rb") as f:
            try:
                content = f.read()
            except Exception as e:
                raise OSError("Unable to load file '{}'.".format(model_path)) from e
            try:
                graph_def.ParseFromString(content)
            except Exception as e:
                raise RuntimeError(
                    "Unable to parse file '{}'.".format(model_path)
                ) from e
            tf.import_graph_def(graph_def, name="")
        input_names = inputs_without_resource(sess, input_names)
        frozen_graph = freeze_session(
            sess, input_names=input_names, output_names=output_names
        )
        frozen_graph = remove_redundant_nodes(frozen_graph, input_names, output_names)
        input_names = remove_redundant_inputs(frozen_graph, input_names)

    tf_reset_default_graph()
    return frozen_graph, input_names, output_names


def from_checkpoint(model_path, input_names, output_names):
    """Load tensorflow graph from checkpoint."""
    # make sure we start with clean default graph
    tf_reset_default_graph()
    with tf_session() as sess:
        saver = tf_import_meta_graph(model_path, clear_devices=True)
        # restore from model_path minus the ".meta"
        saver.restore(sess, model_path[:-5])
        input_names = inputs_without_resource(sess, input_names)
        frozen_graph = freeze_session(
            sess, input_names=input_names, output_names=output_names
        )
        input_names = remove_redundant_inputs(frozen_graph, input_names)

    tf_reset_default_graph()
    return frozen_graph, input_names, output_names


def _from_saved_model_v1(sess, model_path, input_names, output_names, tag, signatures):
    """Load tensorflow graph from saved_model."""

    wrn_no_tag = "'--tag' not specified for saved_model. Using --tag serve"
    wrn_empty_tag = "'--tag' value is empty string. Using tag =[[]]"

    if tag is None:
        tag = [tf.saved_model.tag_constants.SERVING]
        logging.warning(wrn_no_tag)

    if tag == "":
        tag = [[]]
        logging.warning(wrn_empty_tag)

    if not isinstance(tag, list):
        tag = [tag]

    if signatures is None:
        signatures = []
    if isinstance(signatures, str):
        signatures = [signatures]

    imported = tf.saved_model.loader.load(sess, tag, model_path)
    if signatures == []:
        for k in imported.signature_def.keys():
            if k.startswith("_"):
                # consider signatures starting with '_' private
                continue
            signatures.append(k)
    try:
        from tensorflow.contrib.saved_model.python.saved_model import (
            signature_def_utils,
        )

        get_signature_def = (
            lambda meta_graph_def, k: signature_def_utils.get_signature_def_by_key(
                meta_graph_def, k
            )
        )
    except ImportError:
        # TF1.2 changed the api
        get_signature_def = lambda meta_graph_def, k: meta_graph_def.signature_def[k]

    name_map = {}
    if input_names is None:
        input_names = []
        for k in signatures:
            inputs_tensor_info = get_signature_def(imported, k).inputs
            for key, input_tensor in inputs_tensor_info.items():
                input_names.append(input_tensor.name)
                name_map[input_tensor.name] = key
        input_names = list(set(input_names))

    if output_names is None:
        output_names = []
        for k in signatures:
            outputs_tensor_info = get_signature_def(imported, k).outputs
            for key, output_tensor in outputs_tensor_info.items():
                output_names.append(output_tensor.name)
                name_map[output_tensor.name] = key
        output_names = list(set(output_names))

    frozen_graph = freeze_session(
        sess, input_names=input_names, output_names=output_names
    )
    frozen_graph = remove_redundant_nodes(frozen_graph, input_names, output_names)
    input_names = remove_redundant_inputs(frozen_graph, input_names)

    new_name_map = {}
    for name in name_map:
        if name in input_names + output_names:
            new_name_map[name] = name_map[name]

    return frozen_graph, input_names, output_names, new_name_map


def _from_saved_model_v2(model_path, input_names, output_names, signature_def=None):
    model = tf.saved_model.load(model_path)
    all_sigs = model.signatures.keys()
    valid_sigs = [s for s in all_sigs if not s.startswith("_")]
    logging.info("Model signature list: {}".format(valid_sigs))
    if not signature_def:
        signature_def = valid_sigs[0]
    assert signature_def in valid_sigs
    concrete_func = model.signatures[signature_def]

    name_map = {}
    if not input_names:
        input_names = []
        no_resource_inputs = [
            tensor
            for tensor in concrete_func.inputs
            if tensor.dtype != tf.dtypes.resource
        ]
        if (
            hasattr(concrete_func, "_arg_keywords")
            and isinstance(concrete_func._arg_keywords, list)
            and len(concrete_func._arg_keywords) == len(no_resource_inputs)
        ):
            for key, tensor in zip(concrete_func._arg_keywords, no_resource_inputs):
                input_names.append(tensor.name)
                name_map[tensor.name] = key
        else:
            input_names = [
                tensor.name
                for tensor in concrete_func.inputs
                if tensor.dtype != tf.dtypes.resource
            ]

    if not output_names:
        output_names = []
        for key, tensor in concrete_func.structured_outputs.items():
            if tensor.dtype == tf.dtypes.resource:
                continue
            output_names.append(tensor.name)
            name_map[tensor.name] = key

    _, frozen_graph = convert_variables_to_constants_v2_as_graph(
        concrete_func, lower_control_flow=False, aggressive_inlining=False
    )
    frozen_graph = remove_redundant_nodes(frozen_graph, input_names, output_names)
    return frozen_graph, input_names, output_names, name_map


def from_saved_model(
    model_path, input_names=None, output_names=None, tag=None, signatures=None
):
    """Load tensorflow graph from saved_model."""
    tf_reset_default_graph()
    if is_tf2():
        return _from_saved_model_v2(
            model_path, input_names, output_names, signature_def=signatures
        )
    else:
        with tf_session() as sess:
            result = _from_saved_model_v1(
                sess, model_path, input_names, output_names, tag, signatures
            )
    tf_reset_default_graph()
    return result


def update_frozen_graph_constant_name(frozen_graph: GraphDef, constant_data_map):
    name_map = {}
    for node in frozen_graph.node:
        if node.op != "Const" or not node.attr["value"].tensor:
            continue
        data = tensor_proto_to_ndarray(node.attr["value"].tensor)
        key = (data.dtype, tuple(data.shape), data.tobytes())
        if key not in constant_data_map:
            continue
        node.attr["value"].tensor.tensor_content = constant_data_map[key][
            "origin_data"
        ].tobytes()
        new_name = constant_data_map[key]["name"]
        name_map[node.name] = new_name
        node.name = new_name

    for node in frozen_graph.node:
        if node.op == "NoOp":
            del node.input[:]
            continue
        new_input = []
        changed = False
        for inp in node.input:
            if inp[-2:] == ":0":
                inp = inp[:-2]
            if inp[0] == "^" and inp[1:] in name_map:
                changed = True
                new_input.append("^" + name_map[inp[1:]] + ":0")
            elif inp in name_map:
                changed = True
                new_input.append(name_map[inp] + ":0")
            else:
                new_input.append(inp)
        if changed:
            del node.input[:]
            node.input.extend(new_input)
    return frozen_graph


def get_model_constant_info(keras_model):
    old_weights = keras_model.get_weights()
    new_weights = []
    constant_data_map = {}
    for i, data in enumerate(old_weights):
        new_data = np.full(data.shape, i + 9845, dtype=data.dtype)
        new_weights.append(new_data)
        key = (new_data.dtype, tuple(new_data.shape), new_data.tobytes())
        assert key not in constant_data_map
        constant_data_map[key] = {"origin_data": data}
    keras_model.set_weights(new_weights)
    for variable in keras_model.variables:
        key = (
            variable.numpy().dtype,
            tuple(variable.numpy().shape),
            variable.numpy().tobytes(),
        )
        assert "name" not in constant_data_map[key]
        name = variable.name
        if name[-2:] == ":0":
            name = name[:-2]
        constant_data_map[key]["name"] = name
    return constant_data_map


def from_keras(model_path, input_names, output_names, keep_keras_weights_name):
    """Load keras model - experimental for now."""
    from tensorflow.python import keras as _keras
    from tensorflow.python.eager import context
    from tensorflow.python.keras.saving import saving_utils as _saving_utils

    # Handles Keras when Eager mode is enabled.
    custom_objects = None
    if context.executing_eagerly():
        _keras.backend.clear_session()
        _keras.backend.set_learning_phase(False)
        keras_model = _keras.models.load_model(model_path, custom_objects)
        constant_data_map = {}
        if keep_keras_weights_name:
            constant_data_map = get_model_constant_info(keras_model)

        function = _saving_utils.trace_model_call(keras_model)
        concrete_func = function.get_concrete_function()

        # allow to pass inputs and outputs from caller if we don't want all of them
        need_redundant = False
        if not input_names:
            input_names = [
                input_tensor.name
                for input_tensor in concrete_func.inputs
                if input_tensor.dtype != tf.dtypes.resource
            ]
        else:
            need_redundant = True
        if not output_names:
            output_names = [
                output_tensor.name
                for output_tensor in concrete_func.outputs
                if output_tensor.dtype != tf.dtypes.resource
            ]

        frozen_graph = from_function(concrete_func, input_names, output_names)
        if keep_keras_weights_name:
            frozen_graph = update_frozen_graph_constant_name(
                frozen_graph, constant_data_map
            )
        if need_redundant:
            frozen_graph = remove_redundant_nodes(
                frozen_graph, input_names, output_names
            )
            input_names = remove_redundant_inputs(frozen_graph, input_names)
    else:
        raise RuntimeError("Just support eagerly mode")
    return frozen_graph, input_names, output_names
