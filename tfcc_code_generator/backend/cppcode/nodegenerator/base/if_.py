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
from backend.cppcode.common import get_symbol_cpp_dtype, symbol_to_cpp_type


class If(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.If)

    @property
    def code(self):
        then_graph_generator = self.model_generator.graph_generators[
            self.node.then_graph_name
        ]
        else_graph_generator = self.model_generator.graph_generators[
            self.node.else_graph_name
        ]

        code = "\n"
        for i, symbol in enumerate(self.outputs):
            code += "    {symbol_type} {output_name};\n".format(
                symbol_type=symbol_to_cpp_type(symbol, False),
                output_name=self.fmt_dict["outputs"][i],
            )

        code += "    if ({inputs[0]})\n".format(**self.fmt_dict)
        code += "    {\n"
        then_cls_name = then_graph_generator.cls_name
        then_cls_object_name = self.name_manager.get_symbol_name(None)
        code += "        {then_cls_name} {then_cls_object_name};\n".format(
            then_cls_name=then_cls_name, then_cls_object_name=then_cls_object_name
        )
        for i, symbol_name in enumerate(then_graph_generator.input_symbol_names):
            code += "        {then_cls_object_name}.set_{symbol_name}({input_name});\n".format(
                then_cls_object_name=then_cls_object_name,
                symbol_name=symbol_name,
                input_name=self.fmt_dict["inputs"][i + 1],
            )
        code += (
            "        auto s_loop_result = {then_cls_object_name}.process();\n".format(
                then_cls_object_name=then_cls_object_name
            )
        )
        for i in range(len(self.node.output_names)):
            code += "        {output_name} = std::move(std::get<{i}>(s_loop_result));\n".format(
                output_name=self.fmt_dict["outputs"][i], i=i
            )
        code += "    }\n"
        code += "    else\n"
        code += "    {\n"
        else_cls_name = else_graph_generator.cls_name
        else_cls_object_name = self.name_manager.get_symbol_name(None)
        code += "        {else_cls_name} {else_cls_object_name};\n".format(
            else_cls_name=else_cls_name, else_cls_object_name=else_cls_object_name
        )
        for i, symbol_name in enumerate(else_graph_generator.input_symbol_names):
            code += "        {else_cls_object_name}.set_{symbol_name}({input_name});\n".format(
                else_cls_object_name=else_cls_object_name,
                symbol_name=symbol_name,
                input_name=self.fmt_dict["inputs"][
                    i + 1 + self.node.then_graph_capture_count
                ],
            )
        code += (
            "        auto s_loop_result = {else_cls_object_name}.process();\n".format(
                else_cls_object_name=else_cls_object_name
            )
        )
        for i in range(len(self.node.output_names)):
            code += "        {output_name} = std::move(std::get<{i}>(s_loop_result));\n".format(
                output_name=self.fmt_dict["outputs"][i], i=i
            )
        code += "    }\n"

        return code
