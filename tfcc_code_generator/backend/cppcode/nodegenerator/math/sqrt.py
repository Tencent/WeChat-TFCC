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


class Sqrt(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.math.Sqrt)

    @property
    def code(self):
        if self.outputs[0].is_tensor():
            return "auto {outputs[0]} = static_cast<{outputs[0].dtype}>(1) / tfcc::math::rsqrt({inputs[0]});\n".format(
                **self.fmt_dict
            )
        elif self.outputs[0].is_value():
            return "auto {outputs[0]} = static_cast<{outputs[0].dtype}>(std::sqrt({inputs[0]}));\n".format(
                **self.fmt_dict
            )
        else:
            raise RuntimeError("Unknow error")
