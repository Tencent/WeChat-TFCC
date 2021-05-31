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


class Unsqueeze(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.Unsqueeze)

    @property
    def code(self):
        shape_code_list = []
        pos = 0
        for i in range(len(self.outputs[0].shape)):
            if i in self.node.axes:
                shape_code_list.append("1u")
            else:
                symbol_name = self.name_manager.get_symbol_name(self.inputs[0])
                shape_code_list.append("{}.shape({})".format(symbol_name, pos))
                pos += 1
        return "tfcc::View<{outputs[0].dtype}> {outputs[0]}({inputs[0]}, {shape});\n".format(
            shape="{" + ", ".join(shape_code_list) + "}", **self.fmt_dict
        )
