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

# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tfcc_runtime/proto/common.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="tfcc_runtime/proto/common.proto",
    package="tfcc.runtime.common",
    syntax="proto3",
    serialized_pb=_b(
        '\n\x1ftfcc_runtime/proto/common.proto\x12\x13tfcc.runtime.common"o\n\x05Value\x12\x15\n\x0bint64_value\x18\x01 \x01(\x03H\x00\x12\x16\n\x0cuint64_value\x18\x02 \x01(\x04H\x00\x12\x15\n\x0b\x66loat_value\x18\x03 \x01(\x02H\x00\x12\x16\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00\x42\x08\n\x06source*\xa0\x01\n\x08\x44\x61taType\x12\n\n\x06UNKNOW\x10\x00\x12\t\n\x05\x46LOAT\x10\x01\x12\t\n\x05UINT8\x10\x02\x12\x08\n\x04INT8\x10\x03\x12\n\n\x06UINT16\x10\x04\x12\t\n\x05INT16\x10\x05\x12\t\n\x05INT32\x10\x06\x12\t\n\x05INT64\x10\x07\x12\x08\n\x04\x42OOL\x10\t\x12\n\n\x06\x44OUBLE\x10\x0b\x12\n\n\x06UINT32\x10\x0c\x12\n\n\x06UINT64\x10\r\x12\r\n\tCOMPLEX64\x10\x0e\x62\x06proto3'
    ),
)

_DATATYPE = _descriptor.EnumDescriptor(
    name="DataType",
    full_name="tfcc.runtime.common.DataType",
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name="UNKNOW", index=0, number=0, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="FLOAT", index=1, number=1, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="UINT8", index=2, number=2, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="INT8", index=3, number=3, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="UINT16", index=4, number=4, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="INT16", index=5, number=5, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="INT32", index=6, number=6, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="INT64", index=7, number=7, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="BOOL", index=8, number=9, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="DOUBLE", index=9, number=11, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="UINT32", index=10, number=12, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="UINT64", index=11, number=13, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="COMPLEX64", index=12, number=14, options=None, type=None
        ),
    ],
    containing_type=None,
    options=None,
    serialized_start=170,
    serialized_end=330,
)
_sym_db.RegisterEnumDescriptor(_DATATYPE)

DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
UNKNOW = 0
FLOAT = 1
UINT8 = 2
INT8 = 3
UINT16 = 4
INT16 = 5
INT32 = 6
INT64 = 7
BOOL = 9
DOUBLE = 11
UINT32 = 12
UINT64 = 13
COMPLEX64 = 14


_VALUE = _descriptor.Descriptor(
    name="Value",
    full_name="tfcc.runtime.common.Value",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="int64_value",
            full_name="tfcc.runtime.common.Value.int64_value",
            index=0,
            number=1,
            type=3,
            cpp_type=2,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="uint64_value",
            full_name="tfcc.runtime.common.Value.uint64_value",
            index=1,
            number=2,
            type=4,
            cpp_type=4,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="float_value",
            full_name="tfcc.runtime.common.Value.float_value",
            index=2,
            number=3,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="double_value",
            full_name="tfcc.runtime.common.Value.double_value",
            index=3,
            number=4,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[
        _descriptor.OneofDescriptor(
            name="source",
            full_name="tfcc.runtime.common.Value.source",
            index=0,
            containing_type=None,
            fields=[],
        ),
    ],
    serialized_start=56,
    serialized_end=167,
)

_VALUE.oneofs_by_name["source"].fields.append(_VALUE.fields_by_name["int64_value"])
_VALUE.fields_by_name["int64_value"].containing_oneof = _VALUE.oneofs_by_name["source"]
_VALUE.oneofs_by_name["source"].fields.append(_VALUE.fields_by_name["uint64_value"])
_VALUE.fields_by_name["uint64_value"].containing_oneof = _VALUE.oneofs_by_name["source"]
_VALUE.oneofs_by_name["source"].fields.append(_VALUE.fields_by_name["float_value"])
_VALUE.fields_by_name["float_value"].containing_oneof = _VALUE.oneofs_by_name["source"]
_VALUE.oneofs_by_name["source"].fields.append(_VALUE.fields_by_name["double_value"])
_VALUE.fields_by_name["double_value"].containing_oneof = _VALUE.oneofs_by_name["source"]
DESCRIPTOR.message_types_by_name["Value"] = _VALUE
DESCRIPTOR.enum_types_by_name["DataType"] = _DATATYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Value = _reflection.GeneratedProtocolMessageType(
    "Value",
    (_message.Message,),
    dict(
        DESCRIPTOR=_VALUE,
        __module__="tfcc_runtime.proto.common_pb2"
        # @@protoc_insertion_point(class_scope:tfcc.runtime.common.Value)
    ),
)
_sym_db.RegisterMessage(Value)


# @@protoc_insertion_point(module_scope)