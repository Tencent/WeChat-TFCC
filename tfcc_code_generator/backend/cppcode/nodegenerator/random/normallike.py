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


class NormalLike(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.random.NormalLike)

    @property
    def code(self):
        return "auto {outputs[0]} = tfcc::random::normal(tfcc::Shape({inputs[0]}), {mean}, {scale});\n".format(
            mean=get_cpp_constant_value(self.node.mean, self.outputs[0].dtype),
            scale=get_cpp_constant_value(self.node.scale, self.outputs[0].dtype),
            **self.fmt_dict
        )
