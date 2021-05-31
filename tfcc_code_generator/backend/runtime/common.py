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

import ir.framework
from .proto import common_pb2


def data_type_to_proto(dtype: ir.framework.DataType):
    if dtype == ir.framework.DataType.FLOAT:
        return common_pb2.DataType.FLOAT
    elif dtype == ir.framework.DataType.UINT8:
        return common_pb2.DataType.UINT8
    elif dtype == ir.framework.DataType.INT8:
        return common_pb2.DataType.INT8
    elif dtype == ir.framework.DataType.UINT16:
        return common_pb2.DataType.UINT16
    elif dtype == ir.framework.DataType.INT16:
        return common_pb2.DataType.INT16
    elif dtype == ir.framework.DataType.INT32:
        return common_pb2.DataType.INT32
    elif dtype == ir.framework.DataType.INT64:
        return common_pb2.DataType.INT64
    elif dtype == ir.framework.DataType.BOOL:
        return common_pb2.DataType.BOOL
    elif dtype == ir.framework.DataType.DOUBLE:
        return common_pb2.DataType.DOUBLE
    elif dtype == ir.framework.DataType.UINT32:
        return common_pb2.DataType.UINT32
    elif dtype == ir.framework.DataType.UINT64:
        return common_pb2.DataType.UINT64
    elif dtype == ir.framework.DataType.COMPLEX64:
        return common_pb2.DataType.COMPLEX64
    else:
        raise RuntimeError("Unknow data type {}".format(dtype))
