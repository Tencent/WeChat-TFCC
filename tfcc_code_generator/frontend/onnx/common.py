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

import typing
import numpy as np
import onnx
import ir.framework


def onnx_tensor_to_symbol(tensor: onnx.TensorProto, name: str = None):
    def transfer_tensor_data(tensor: onnx.TensorProto, dtype):
        if tensor.float_data:
            return np.asarray(list(tensor.float_data), dtype=dtype)
        elif tensor.int32_data:
            return np.asarray(list(tensor.int32_data), dtype=dtype)
        elif tensor.string_data:
            return np.asarray(tensor.string_data, dtype=dtype)
        elif tensor.int64_data:
            return np.asarray(list(tensor.int64_data), dtype=dtype)
        elif not tensor.raw_data:
            return np.asarray([], dtype=dtype)
        else:
            return np.frombuffer(tensor.raw_data, dtype=dtype)

    if tensor.data_type == onnx.TensorProto.FLOAT:
        data = transfer_tensor_data(tensor, np.float32)
    elif tensor.data_type == onnx.TensorProto.DOUBLE:
        data = transfer_tensor_data(tensor, np.float64)
    elif tensor.data_type == onnx.TensorProto.UINT8:
        data = transfer_tensor_data(tensor, np.uint8)
    elif tensor.data_type == onnx.TensorProto.INT8:
        data = transfer_tensor_data(tensor, np.int8)
    elif tensor.data_type == onnx.TensorProto.UINT16:
        data = transfer_tensor_data(tensor, np.uint16)
    elif tensor.data_type == onnx.TensorProto.INT16:
        data = transfer_tensor_data(tensor, np.int16)
    elif tensor.data_type == onnx.TensorProto.UINT32:
        data = transfer_tensor_data(tensor, np.uint32)
    elif tensor.data_type == onnx.TensorProto.INT32:
        data = transfer_tensor_data(tensor, np.int32)
    elif tensor.data_type == onnx.TensorProto.UINT64:
        data = transfer_tensor_data(tensor, np.uint64)
    elif tensor.data_type == onnx.TensorProto.INT64:
        data = transfer_tensor_data(tensor, np.int64)
    elif tensor.data_type == onnx.TensorProto.BOOL:
        data = transfer_tensor_data(tensor, np.bool)
    elif tensor.data_type == onnx.TensorProto.STRING:
        data = transfer_tensor_data(tensor, np.str_)
    else:
        raise RuntimeError("unsupport data type: {}".format(tensor.data_type))
    if tensor.dims:
        shape = list(tensor.dims)
        data = data.reshape(shape)
    else:
        shape = [1]

    dtype = ir.framework.DataType(tensor.data_type)

    if not name:
        name = tensor.name

    assert name

    symbol = ir.framework.Symbol(name)
    symbol.dtype = dtype
    if tensor.dims:
        symbol.stype = ir.framework.SymbolType.CONSTANT_TENSOR
    elif data.size == 1:
        symbol.stype = ir.framework.SymbolType.CONSTANT_VALUE
    else:
        symbol.stype = ir.framework.SymbolType.CONSTANT_VECTOR
    symbol.shape = shape
    symbol.data = data
    symbol.origin_stype = symbol.stype

    return symbol


def onnx_value_info_to_symbol(tensor: onnx.ValueInfoProto):
    assert tensor.type.tensor_type
    shape = []
    for s in tensor.type.tensor_type.shape.dim:
        if s.dim_param:
            shape.append(s.dim_param)
        elif s.dim_value:
            shape.append(s.dim_value)
        else:
            raise RuntimeError("Unknow shape format")
    symbol = ir.framework.symbol.Symbol(tensor.name)
    symbol.dtype = tensor.type.tensor_type.elem_type
    symbol.shape = shape
    return symbol


def parse_onnx_attribute(attr: onnx.AttributeProto):
    name = attr.name
    value = None
    if attr.type == onnx.AttributeProto.FLOAT:
        value = attr.f
    elif attr.type == onnx.AttributeProto.INT:
        value = attr.i
    elif attr.type == onnx.AttributeProto.STRING:
        value = attr.s
    elif attr.type == onnx.AttributeProto.TENSOR:
        value = attr.t
    elif attr.type == onnx.AttributeProto.GRAPH:
        value = attr.g
    elif attr.type == onnx.AttributeProto.SPARSE_TENSOR:
        value = attr.sparse_tensor
    elif attr.type == onnx.AttributeProto.FLOATS:
        value = list(attr.floats)
    elif attr.type == onnx.AttributeProto.INTS:
        value = list(attr.ints)
    elif attr.type == onnx.AttributeProto.STRINGS:
        value = list(attr.strings)
    elif attr.type == onnx.AttributeProto.TENSORS:
        value = list(attr.tensors)
    elif attr.type == onnx.AttributeProto.GRAPHS:
        value = list(attr.graphs)
    elif attr.type == onnx.AttributeProto.SPARSE_TENSORS:
        value = list(attr.sparse_tensors)
    return name, value
