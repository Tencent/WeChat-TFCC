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
from backend.cppcode.common import get_symbol_cpp_dtype


class BidirectionalLSTM(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.rnn.BidirectionalLSTM)

    @property
    def code(self):
        code = "tfcc::Variable<{dtype}> {outputs[0]}, {outputs[1]}, {outputs[2]}, {outputs[3]}, {outputs[4]}, {outputs[5]};\n".format(
            dtype=get_symbol_cpp_dtype(self.outputs[0]), **self.fmt_dict
        )
        code += """
    std::tie({outputs[0]}, {outputs[1]}, {outputs[2]}, {outputs[3]}, {outputs[4]}, {outputs[5]}) = tfcc::helper::rnn::bidirectional_lstm(
        {inputs[0]},
        {inputs[1]},
        {inputs[2]},
        {inputs[3]},
        {inputs[4]},
        {inputs[5]}
    );
""".format(
            **self.fmt_dict
        )
        return code
