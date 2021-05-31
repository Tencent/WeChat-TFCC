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


class Reduce(NodeGenerator):
    _reduceName = None
    _reduceCls = None

    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, cls._reduceCls)

    @property
    def code(self):
        if set(self.node.axes) == set(
            range(min(self.node.axes), len(self.inputs[0].shape))
        ):
            reduce_code = "auto {outputs[0]} = tfcc::math::{reduce_name}({inputs[0]}, {keep});\n".format(
                reduce_name=self._reduceName, keep=min(self.node.axes), **self.fmt_dict
            )
        else:
            reduce_code = "auto {outputs[0]} = tfcc::helper::math::{reduce_name}({inputs[0]}, {axes});\n".format(
                reduce_name=self._reduceName,
                axes="{" + ", ".join([str(x) for x in self.node.axes]) + "}",
                **self.fmt_dict
            )
        return reduce_code


class ReduceMean(Reduce):
    _reduceName = "reduce_mean"
    _reduceCls = ir.node.math.ReduceMean


class ReduceSum(Reduce):
    _reduceName = "reduce_sum"
    _reduceCls = ir.node.math.ReduceSum


class ReduceProd(Reduce):
    _reduceName = "reduce_prod"
    _reduceCls = ir.node.math.ReduceProd


class ReduceMax(Reduce):
    _reduceName = "reduce_max"
    _reduceCls = ir.node.math.ReduceMax


class ReduceMin(Reduce):
    _reduceName = "reduce_min"
    _reduceCls = ir.node.math.ReduceMin
