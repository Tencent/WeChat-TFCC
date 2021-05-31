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
import onnx
import onnx.defs
from frontend.onnx.common import parse_onnx_attribute

_schemas = None


def _get_schema(domain: str, op_type: str, version: int):
    global _schemas
    if not _schemas:
        schemas = {}
        for schema in onnx.defs.get_all_schemas_with_history():
            if schema.name not in schemas:
                schemas[schema.name] = []
            schemas[schema.name].append(schema)
        _schemas = schemas

    result = None

    for schema in _schemas[op_type]:
        if schema.domain != domain:
            continue
        if schema.since_version > version:
            continue
        if not result or result.since_version < schema.since_version:
            result = schema
    return result


class Converter(object):
    def __init__(self, op_set: dict):
        self._schema = _get_schema(
            self.domain, self.op_type, op_set[self.domain].version
        )
        self._valid = True
        if self._schema:
            if self.accept_versions and not self.since_version in self.accept_versions:
                self._valid = False

    @property
    def valid(self):
        return self._valid

    @property
    def domain(self) -> str:
        return ""

    @property
    def op_type(self) -> str:
        return self.__class__.__name__.split(".")[-1]

    @property
    def schema(self):
        return self._schema

    @property
    def since_version(self) -> int:
        return self.schema.since_version

    @property
    def min_input_count(self) -> int:
        return self.schema.min_input

    @property
    def max_input_count(self) -> int:
        return self.schema.max_input

    @property
    def min_output_count(self) -> int:
        return self.schema.min_output

    @property
    def max_output_count(self) -> int:
        return self.schema.max_output

    @property
    def accept_versions(self) -> set:
        raise NotImplementedError

    def accept(self, node_proto: onnx.NodeProto):
        if not self.valid:
            return False
        if not self._schema:
            return False
        if node_proto.domain != self.domain:
            return False
        if node_proto.op_type != self.op_type:
            return False
        if (
            len(node_proto.input) < self.min_input_count
            or len(node_proto.input) > self.max_input_count
        ):
            return False
        if (
            len(node_proto.output) < self.min_output_count
            or len(node_proto.output) > self.max_output_count
        ):
            return False
        return True

    def get_attributes(self, node_proto: onnx.NodeProto):
        attributes = {}
        for attribute in node_proto.attribute:
            name, value = parse_onnx_attribute(attribute)
            attributes[name] = value

        for attribute in self.schema.attributes.values():
            if attribute.name not in attributes:
                if attribute.required:
                    raise RuntimeError(
                        "node: {} loss required attribute {}",
                        node_proto,
                        attribute.name,
                    )
                if attribute.default_value.name:
                    name, value = parse_onnx_attribute(attribute.default_value)
                    attributes[name] = value
        return attributes
