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


class BatchNormalization(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.nn.BatchNormalization)

    @property
    def code(self):
        return "auto {outputs[0]} = tfcc::nn::batch_normalization({inputs[0]}, {node.axis}, {inputs[1]}, {inputs[2]}, {inputs[3]}, {inputs[4]}, {epsilon});\n".format(
            epsilon=get_cpp_constant_value(self.node.epsilon, self.outputs[0].dtype),
            **self.fmt_dict
        )
