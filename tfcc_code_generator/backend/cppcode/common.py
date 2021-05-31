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


def get_cpp_constant_value(value, dtype: ir.framework.DataType):
    if dtype == ir.framework.DataType.FLOAT:
        ctype = "float"
    elif dtype == ir.framework.DataType.DOUBLE:
        ctype = "double"
    elif dtype == ir.framework.DataType.UINT8:
        ctype = "uint8_t"
    elif dtype == ir.framework.DataType.INT8:
        ctype = "int8_t"
    elif dtype == ir.framework.DataType.UINT16:
        ctype = "uint16_t"
    elif dtype == ir.framework.DataType.INT16:
        ctype = "int16_t"
    elif dtype == ir.framework.DataType.UINT32:
        ctype = "uint32_t"
    elif dtype == ir.framework.DataType.INT32:
        ctype = "int32_t"
    elif dtype == ir.framework.DataType.UINT64:
        ctype = "uint64_t"
    elif dtype == ir.framework.DataType.INT64:
        ctype = "int64_t"
    elif dtype == ir.framework.DataType.BOOL:
        ctype = "bool"
    else:
        raise RuntimeError("Unknow dtype")

    if dtype == ir.framework.DataType.INT64:
        return "static_cast<{dtype}>({value}l)".format(dtype=ctype, value=value)
    elif dtype == ir.framework.DataType.UINT64:
        return "static_cast<{dtype}>({value}lu)".format(dtype=ctype, value=value)
    elif dtype == ir.framework.DataType.STRING:
        return '"{value}"'.format(value=value.replace("\\", "\\\\"))
    elif dtype == ir.framework.DataType.BOOL:
        return "true" if value else "false"
    else:
        if value == float("inf"):
            return "std::numeric_limits<{dtype}>::infinity()".format(dtype=ctype)
        elif value == float("-inf"):
            return "-std::numeric_limits<{dtype}>::infinity()".format(dtype=ctype)
        return "static_cast<{dtype}>({value})".format(dtype=ctype, value=value)


def get_symbol_cpp_dtype(symbol: ir.framework.Symbol):
    if symbol.dtype == ir.framework.DataType.FLOAT:
        dtype = "float"
    elif symbol.dtype == ir.framework.DataType.DOUBLE:
        dtype = "double"
    elif symbol.dtype == ir.framework.DataType.UINT8:
        dtype = "uint8_t"
    elif symbol.dtype == ir.framework.DataType.INT8:
        dtype = "int8_t"
    elif symbol.dtype == ir.framework.DataType.UINT16:
        dtype = "uint16_t"
    elif symbol.dtype == ir.framework.DataType.INT16:
        dtype = "int16_t"
    elif symbol.dtype == ir.framework.DataType.UINT32:
        dtype = "uint32_t"
    elif symbol.dtype == ir.framework.DataType.INT32:
        dtype = "int32_t"
    elif symbol.dtype == ir.framework.DataType.UINT64:
        dtype = "uint64_t"
    elif symbol.dtype == ir.framework.DataType.INT64:
        dtype = "int64_t"
    elif symbol.dtype == ir.framework.DataType.BOOL:
        if symbol.is_tensor():
            dtype = "uint8_t"
        else:
            dtype = "bool"
    else:
        raise RuntimeError("Unknow dtype")
    return dtype


def symbol_to_cpp_type(symbol: ir.framework.Symbol, ref: bool, copyable: bool = False):
    dtype = get_symbol_cpp_dtype(symbol)
    if not copyable:
        if symbol.is_tensor() and ref:
            return "tfcc::Tensor<{dtype}>".format(dtype=dtype)
        elif symbol.is_tensor() and not ref:
            if symbol.stype == ir.framework.SymbolType.CONSTANT_TENSOR:
                return "tfcc::View<{dtype}>".format(dtype=dtype)
            elif symbol.stype == ir.framework.SymbolType.VIEW:
                return "tfcc::View<{dtype}>".format(dtype=dtype)
            elif symbol.stype == ir.framework.SymbolType.VARIABLE:
                return "tfcc::Variable<{dtype}>".format(dtype=dtype)
            else:
                raise RuntimeError("Unknow stype")
    else:
        assert ref == False
        if symbol.is_tensor():
            if symbol.stype == ir.framework.SymbolType.CONSTANT_TENSOR:
                return "tfcc::Variable<{dtype}>".format(dtype=dtype)
            elif symbol.stype == ir.framework.SymbolType.VIEW:
                return "tfcc::Variable<{dtype}>".format(dtype=dtype)
            elif symbol.stype == ir.framework.SymbolType.VARIABLE:
                return "tfcc::Variable<{dtype}>".format(dtype=dtype)
            else:
                raise RuntimeError("Unknow stype")
    if symbol.is_value():
        return "{dtype}".format(dtype=dtype)
    elif symbol.is_vector():
        return "std::vector<{dtype}>".format(dtype=dtype)
