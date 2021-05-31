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

import ir.node
from backend.cppcode.nodegenerator.nodegenerator import NodeGenerator


class Equal(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.relation.Equal)

    @property
    def code(self):
        if (
            self.inputs[0].is_value()
            and self.inputs[1].is_tensor()
            and self.inputs[0].dtype == ir.framework.DataType.BOOL
        ):
            return "auto {outputs[0]} = tfcc::relation::equal(static_cast<uint8_t>({inputs[0]}), {inputs[1]});\n".format(
                **self.fmt_dict
            )
        elif (
            self.inputs[0].is_tensor()
            and self.inputs[1].is_value()
            and self.inputs[1].dtype == ir.framework.DataType.BOOL
        ):
            return "auto {outputs[0]} = tfcc::relation::equal({inputs[0]}, static_cast<uint8_t>({inputs[1]}));\n".format(
                **self.fmt_dict
            )
        elif self.inputs[0].is_value() and self.inputs[1].is_value():
            return "bool {outputs[0]} = {inputs[0]} == {inputs[1]};\n".format(
                **self.fmt_dict
            )
        return "auto {outputs[0]} = tfcc::relation::equal({inputs[0]}, {inputs[1]});\n".format(
            **self.fmt_dict
        )
