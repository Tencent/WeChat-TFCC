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


class Split(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.Split)

    @property
    def code(self):
        fmt = ""
        for i in range(len(self.outputs)):
            if i > 0:
                fmt += "    "
            fmt += (
                "auto {outputs["
                + str(i)
                + "]} = tfcc::base::slice({inputs[0]}, {node.axis}, {inputs[0]}.shape({node.axis}) / {count} * "
                + str(i)
            )
            fmt += ", {inputs[0]}.shape({node.axis}) / {count} * " + str(i + 1) + ");\n"
        return fmt.format(count=len(self.outputs), **self.fmt_dict)
