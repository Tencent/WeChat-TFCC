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

from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node
from ..converter import Converter


# implement for https://tensorflow.google.cn/api_docs/python/tf/raw_ops/FusedBatchNormV3?hl=en
class FusedBatchNormV3(Converter):
    def __call__(self, op: Operation, graph: ir.framework.Graph):
        if not self.accept(op):
            return False

        input_name, scale_name, offset_name, mean_name, variance_name = [
            inp.name for inp in op.inputs
        ]
        oup_strs = [oup.name for oup in op.outputs]
        if len(oup_strs) == 0:
            return False
        output_name = oup_strs[0]

        data_format = op.get_attr("data_format")
        epsilon = op.get_attr("epsilon")

        if data_format != b"NHWC":
            return False
        input_names = [input_name, scale_name, offset_name, mean_name, variance_name]
        axis = len(graph.get_symbol(input_name).shape) - 1
        graph.append_node(
            ir.node.nn.BatchNormalization(
                op.name, graph, input_names, [output_name], axis=axis, epsilon=epsilon
            )
        )
        return True
