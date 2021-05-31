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
from backend.cppcode.common import get_cpp_constant_value


class Flatten(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.Flatten)

    @property
    def code(self):
        str_s0 = ["1u"]
        for i in range(self.node.axis):
            str_s0.append("{inputs[0]}.shape({i})".format(i=i, **self.fmt_dict))
        str_s1 = ["1u"]
        for i in range(self.node.axis, len(self.inputs[0].shape)):
            str_s1.append("{inputs[0]}.shape({i})".format(i=i, **self.fmt_dict))
        shape = "{" + " * ".join(str_s0) + ", " + " * ".join(str_s1) + "}"
        return "tfcc::View<{outputs[0].dtype}> {outputs[0]}({inputs[0]}, tfcc::Shape({shape}));\n".format(
            shape=shape, **self.fmt_dict
        )
