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
# source: tfcc_runtime/proto/operations/random.proto

import sys

_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode("latin1"))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from .. import common_pb2 as tfcc__runtime_dot_proto_dot_common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
    name="tfcc_runtime/proto/operations/random.proto",
    package="tfcc.runtime.operations.random",
    syntax="proto3",
    serialized_pb=_b(
        '\n*tfcc_runtime/proto/operations/random.proto\x12\x1etfcc.runtime.operations.random\x1a\x1ftfcc_runtime/proto/common.proto"\xb4\x01\n\nNormalLike\x12(\n\x04mean\x18\x01 \x01(\x0b\x32\x1a.tfcc.runtime.common.Value\x12)\n\x05scale\x18\x02 \x01(\x0b\x32\x1a.tfcc.runtime.common.Value\x12\x30\n\tdata_type\x18\x03 \x01(\x0e\x32\x1d.tfcc.runtime.common.DataType"\x1f\n\x07VERSION\x12\x05\n\x01_\x10\x00\x12\r\n\tVERSION_1\x10\x01"\xb3\x01\n\x0bUniformLike\x12\'\n\x03low\x18\x01 \x01(\x0b\x32\x1a.tfcc.runtime.common.Value\x12(\n\x04high\x18\x02 \x01(\x0b\x32\x1a.tfcc.runtime.common.Value\x12\x30\n\tdata_type\x18\x03 \x01(\x0e\x32\x1d.tfcc.runtime.common.DataType"\x1f\n\x07VERSION\x12\x05\n\x01_\x10\x00\x12\r\n\tVERSION_1\x10\x01\x62\x06proto3'
    ),
    dependencies=[
        tfcc__runtime_dot_proto_dot_common__pb2.DESCRIPTOR,
    ],
)


_NORMALLIKE_VERSION = _descriptor.EnumDescriptor(
    name="VERSION",
    full_name="tfcc.runtime.operations.random.NormalLike.VERSION",
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name="_", index=0, number=0, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="VERSION_1", index=1, number=1, options=None, type=None
        ),
    ],
    containing_type=None,
    options=None,
    serialized_start=261,
    serialized_end=292,
)
_sym_db.RegisterEnumDescriptor(_NORMALLIKE_VERSION)

_UNIFORMLIKE_VERSION = _descriptor.EnumDescriptor(
    name="VERSION",
    full_name="tfcc.runtime.operations.random.UniformLike.VERSION",
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name="_", index=0, number=0, options=None, type=None
        ),
        _descriptor.EnumValueDescriptor(
            name="VERSION_1", index=1, number=1, options=None, type=None
        ),
    ],
    containing_type=None,
    options=None,
    serialized_start=261,
    serialized_end=292,
)
_sym_db.RegisterEnumDescriptor(_UNIFORMLIKE_VERSION)


_NORMALLIKE = _descriptor.Descriptor(
    name="NormalLike",
    full_name="tfcc.runtime.operations.random.NormalLike",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="mean",
            full_name="tfcc.runtime.operations.random.NormalLike.mean",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="scale",
            full_name="tfcc.runtime.operations.random.NormalLike.scale",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="data_type",
            full_name="tfcc.runtime.operations.random.NormalLike.data_type",
            index=2,
            number=3,
            type=14,
            cpp_type=8,
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _NORMALLIKE_VERSION,
    ],
    options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=112,
    serialized_end=292,
)


_UNIFORMLIKE = _descriptor.Descriptor(
    name="UniformLike",
    full_name="tfcc.runtime.operations.random.UniformLike",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name="low",
            full_name="tfcc.runtime.operations.random.UniformLike.low",
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="high",
            full_name="tfcc.runtime.operations.random.UniformLike.high",
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None,
            file=DESCRIPTOR,
        ),
        _descriptor.FieldDescriptor(
            name="data_type",
            full_name="tfcc.runtime.operations.random.UniformLike.data_type",
            index=2,
            number=3,
            type=14,
            cpp_type=8,
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
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _UNIFORMLIKE_VERSION,
    ],
    options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=295,
    serialized_end=474,
)

_NORMALLIKE.fields_by_name[
    "mean"
].message_type = tfcc__runtime_dot_proto_dot_common__pb2._VALUE
_NORMALLIKE.fields_by_name[
    "scale"
].message_type = tfcc__runtime_dot_proto_dot_common__pb2._VALUE
_NORMALLIKE.fields_by_name[
    "data_type"
].enum_type = tfcc__runtime_dot_proto_dot_common__pb2._DATATYPE
_NORMALLIKE_VERSION.containing_type = _NORMALLIKE
_UNIFORMLIKE.fields_by_name[
    "low"
].message_type = tfcc__runtime_dot_proto_dot_common__pb2._VALUE
_UNIFORMLIKE.fields_by_name[
    "high"
].message_type = tfcc__runtime_dot_proto_dot_common__pb2._VALUE
_UNIFORMLIKE.fields_by_name[
    "data_type"
].enum_type = tfcc__runtime_dot_proto_dot_common__pb2._DATATYPE
_UNIFORMLIKE_VERSION.containing_type = _UNIFORMLIKE
DESCRIPTOR.message_types_by_name["NormalLike"] = _NORMALLIKE
DESCRIPTOR.message_types_by_name["UniformLike"] = _UNIFORMLIKE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

NormalLike = _reflection.GeneratedProtocolMessageType(
    "NormalLike",
    (_message.Message,),
    dict(
        DESCRIPTOR=_NORMALLIKE,
        __module__="tfcc_runtime.proto.operations.random_pb2"
        # @@protoc_insertion_point(class_scope:tfcc.runtime.operations.random.NormalLike)
    ),
)
_sym_db.RegisterMessage(NormalLike)

UniformLike = _reflection.GeneratedProtocolMessageType(
    "UniformLike",
    (_message.Message,),
    dict(
        DESCRIPTOR=_UNIFORMLIKE,
        __module__="tfcc_runtime.proto.operations.random_pb2"
        # @@protoc_insertion_point(class_scope:tfcc.runtime.operations.random.UniformLike)
    ),
)
_sym_db.RegisterMessage(UniformLike)


# @@protoc_insertion_point(module_scope)
