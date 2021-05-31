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


class Stack(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.Stack)

    @property
    def code(self):
        name_list = ["&" + self.name_manager.get_symbol_name(x) for x in self.inputs]
        return (
            "auto {outputs[0]} = tfcc::base::stack({{{vals}}}, {node.axis});\n".format(
                vals=", ".join(name_list), **self.fmt_dict
            )
        )