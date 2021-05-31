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
from distutils.version import LooseVersion

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import tensor_util

from . import utils


def counter():
    count = 0

    def _counter():
        nonlocal count
        count += 1
        return count

    return _counter


tensor_shape_unknown_counter = counter()


def get_tf_version():
    return LooseVersion(tf.__version__)


def get_tf_tensor_shape(tensor):
    shape = tensor.get_shape().as_list()
    shape = [
        "unknown{}".format(tensor_shape_unknown_counter()) if x is None else x
        for x in shape
    ]
    return shape


def get_tf_tensor_data(tensor):
    if not isinstance(tensor, tensor_pb2.TensorProto):
        raise ValueError("Require TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    if not isinstance(np_data, np.ndarray):
        raise ValueError("%r isn't ndarray", np_data)
    return np_data


def get_tf_nodedef_by_name(graphdef, name):
    for node in graphdef.node:
        if node.name == utils.node_name(name):
            return node
    return None


def tensor_proto_to_ndarray(tensor: tensor_pb2.TensorProto):
    if tensor.dtype == types_pb2.DT_FLOAT:
        dtype = np.dtype(np.float32)
    elif tensor.dtype == types_pb2.DT_INT32:
        dtype = np.dtype(np.int32)
    elif tensor.dtype == types_pb2.DT_INT64:
        dtype = np.dtype(np.int64)
    else:
        raise RuntimeError("Unknow tensor dtype")
    shape = [dim.size for dim in tensor.tensor_shape.dim]
    if tensor.tensor_content:
        data = np.frombuffer(tensor.tensor_content, dtype=dtype)
    elif tensor.float_val:
        data = np.asarray(tensor.float_val, dtype=dtype)
    elif tensor.double_val:
        data = np.asarray(tensor.double_val, dtype=dtype)
    elif tensor.int_val:
        data = np.asarray(tensor.int_val, dtype=dtype)
    elif tensor.int64_val:
        data = np.asarray(tensor.int64_val, dtype=dtype)
    else:
        raise RuntimeError("Unknow tensor val type")

    data = data.reshape(shape)
    return data
