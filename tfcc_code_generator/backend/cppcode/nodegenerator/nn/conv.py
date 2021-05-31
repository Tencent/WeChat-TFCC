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


class Conv(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.nn.Conv)

    @property
    def code(self):
        if self.node.group == 1:
            conv_attributes = self.node.attributes
            if (
                conv_attributes["dilate_height"] > 1
                or conv_attributes["dilate_width"] > 1
            ):
                return "auto {outputs[0]} = tfcc::nn::conv2d({inputs[0]}, !{nchw}, {inputs[1]}, {node.padding_height}, {node.padding_width}, {node.stride_height}, {node.stride_width}, {node.dilate_height}, {node.dilate_width});\n".format(
                    nchw="true" if self.node.nchw else "false", **self.fmt_dict
                )
            return "auto {outputs[0]} = tfcc::nn::conv2d({inputs[0]}, !{nchw}, {inputs[1]}, {node.padding_height}, {node.padding_width}, {node.stride_height}, {node.stride_width});\n".format(
                nchw="true" if self.node.nchw else "false", **self.fmt_dict
            )
        else:
            conv_attributes = self.node.attributes
            if (
                conv_attributes["dilate_height"] > 1
                or conv_attributes["dilate_width"] > 1
            ):
                return "auto {outputs[0]} = tfcc::nn::conv2d({inputs[0]}, !{nchw}, {inputs[1]}, {node.padding_height}, {node.padding_width}, {node.stride_height}, {node.stride_width}, {node.dilate_height}, {node.dilate_width}, {node.group});\n".format(
                    nchw="true" if self.node.nchw else "false", **self.fmt_dict
                )
            return "auto {outputs[0]} = tfcc::nn::conv2d({inputs[0]}, !{nchw}, {inputs[1]}, {node.padding_height}, {node.padding_width}, {node.stride_height}, {node.stride_width}, 1, 1, {node.group});\n".format(
                nchw="true" if self.node.nchw else "false", **self.fmt_dict
            )
