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
from tensorflow.core.framework import types_pb2
import ir.framework

tf_to_np_dtype = {
    types_pb2.DT_HALF: np.float16,
    types_pb2.DT_FLOAT: np.float32,
    types_pb2.DT_DOUBLE: np.double,
    types_pb2.DT_UINT8: np.uint8,
    types_pb2.DT_INT8: np.int8,
    types_pb2.DT_UINT16: np.uint16,
    types_pb2.DT_INT16: np.int16,
    types_pb2.DT_UINT32: np.uint32,
    types_pb2.DT_INT32: np.int32,
    types_pb2.DT_UINT64: np.uint64,
    types_pb2.DT_INT64: np.int64,
    types_pb2.DT_BOOL: np.bool,
    types_pb2.DT_STRING: "S1",
}

tf_to_symbol_dtype = {
    types_pb2.DT_HALF: ir.framework.DataType.FLOAT,
    types_pb2.DT_FLOAT: ir.framework.DataType.FLOAT,
    types_pb2.DT_DOUBLE: ir.framework.DataType.DOUBLE,
    types_pb2.DT_UINT8: ir.framework.DataType.UINT8,
    types_pb2.DT_INT8: ir.framework.DataType.INT8,
    types_pb2.DT_UINT16: ir.framework.DataType.UINT16,
    types_pb2.DT_INT16: ir.framework.DataType.INT16,
    types_pb2.DT_UINT32: ir.framework.DataType.UINT32,
    types_pb2.DT_INT32: ir.framework.DataType.INT32,
    types_pb2.DT_UINT64: ir.framework.DataType.UINT64,
    types_pb2.DT_INT64: ir.framework.DataType.INT64,
    types_pb2.DT_BOOL: ir.framework.DataType.BOOL,
    types_pb2.DT_STRING: ir.framework.DataType.STRING,
}
