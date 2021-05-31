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


class Loop(NodeGenerator):
    @classmethod
    def accept(cls, node: ir.node.Node):
        return isinstance(node, ir.node.base.Loop)

    @property
    def code(self):
        sub_graph_generator = self.model_generator.graph_generators[
            self.node.sub_graph_name
        ]
        cls_name = sub_graph_generator.cls_name
        cls_object_name = self.name_manager.get_symbol_name(None)
        code = "{cls_name} {cls_object_name};\n".format(
            cls_name=cls_name, cls_object_name=cls_object_name
        )

        # set inputs
        if self.node.capture_count > 0:
            for i, name in enumerate(
                sub_graph_generator.graph.inputs[-self.node.capture_count :]
            ):
                symbol = sub_graph_generator.graph.get_symbol(name)
                symbol_name = sub_graph_generator.name_manager.get_symbol_name(
                    symbol, symbol.name
                )
                code += (
                    "    {cls_object_name}.set_{symbol_name}({input_name});\n".format(
                        cls_object_name=cls_object_name,
                        symbol_name=symbol_name,
                        input_name=self.fmt_dict["inputs"][
                            self.node.carried_count + 1 + i
                        ],
                    )
                )

        condition_name = self.name_manager.get_symbol_name(None)
        code += "    bool {condition_name} = {inputs[0]};\n".format(
            condition_name=condition_name, **self.fmt_dict
        )

        # carried declaration
        for i, symbol in enumerate(self.outputs[: self.node.carried_count]):
            if symbol.is_tensor():
                code += "    tfcc::Variable<{dtype}> {carried_name};\n".format(
                    dtype=get_symbol_cpp_dtype(symbol),
                    carried_name=self.fmt_dict["outputs"][i],
                )
            elif symbol.is_vector():
                code += "    std::vector<{dtype}> {carried_name};\n".format(
                    dtype=get_symbol_cpp_dtype(symbol),
                    carried_name=self.fmt_dict["outputs"][i],
                )
            elif symbol.is_value():
                code += "    {dtype} {carried_name};\n".format(
                    dtype=get_symbol_cpp_dtype(symbol),
                    carried_name=self.fmt_dict["outputs"][i],
                )
            else:
                raise RuntimeError("Unknow error")

        scan_names = []
        for name in sub_graph_generator.graph.outputs[self.node.carried_count + 1 :]:
            symbol = sub_graph_generator.graph.get_symbol(name)
            scan_name = self.name_manager.get_symbol_name(None)
            if symbol.is_tensor():
                code += (
                    "    std::vector<tfcc::Variable<{dtype}>> {scan_name};\n".format(
                        dtype=get_symbol_cpp_dtype(symbol), scan_name=scan_name
                    )
                )
            elif symbol.is_vector():
                code += "    std::vector<{dtype}> {scan_name};\n".format(
                    dtype=get_symbol_cpp_dtype(symbol), scan_name=scan_name
                )
            elif symbol.is_value():
                code += "    std::vector<{dtype}> {scan_name};\n".format(
                    dtype=get_symbol_cpp_dtype(symbol), scan_name=scan_name
                )
            else:
                raise RuntimeError("Unknow error")
            scan_names.append(scan_name)

        # check need loop count
        need_loop_count = False
        for i in range(len(scan_names)):
            symbol = sub_graph_generator.graph.get_symbol(
                sub_graph_generator.graph.outputs[1 + self.node.carried_count + i]
            )
            if symbol.is_vector():
                need_loop_count = True

        if need_loop_count:
            loop_count_name = self.name_manager.get_symbol_name(None)
            code += "    unsigned {loop_count_name} = 0;\n".format(
                loop_count_name=loop_count_name
            )
        if (
            len(self.node.input_names)
            == 2 + self.node.carried_count + self.node.capture_count
        ):
            # has max loop
            code += "    for ({max_loop.dtype} i = 0; i < {max_loop} && {condition_name}; ++i)\n".format(
                condition_name=condition_name,
                max_loop=self.fmt_dict["inputs"][-1],
                **self.fmt_dict
            )
        else:
            code += "    for (unsigned i = 0; {condition_name}; ++i)\n".format(
                condition_name=condition_name, **self.fmt_dict
            )

        code += "    {\n"
        code += "        {cls_object_name}.set_{symbol_name}(i);\n".format(
            cls_object_name=cls_object_name,
            symbol_name=sub_graph_generator.input_symbol_names[0],
        )
        code += (
            "        {cls_object_name}.set_{symbol_name}({condition_name});\n".format(
                cls_object_name=cls_object_name,
                symbol_name=sub_graph_generator.input_symbol_names[1],
                condition_name=condition_name,
            )
        )
        code += "        if (i == 0)\n"
        code += "        {\n"
        for i in range(self.node.carried_count):
            code += "            {cls_object_name}.set_{symbol_name}({input_name});\n".format(
                cls_object_name=cls_object_name,
                symbol_name=sub_graph_generator.input_symbol_names[2 + i],
                input_name=self.fmt_dict["inputs"][1 + i],
            )
        code += "        }\n"
        code += "        else\n"
        code += "        {\n"
        for i in range(self.node.carried_count):
            code += "            {cls_object_name}.set_{symbol_name}({carried_name});\n".format(
                cls_object_name=cls_object_name,
                symbol_name=sub_graph_generator.input_symbol_names[2 + i],
                carried_name=self.fmt_dict["outputs"][i],
            )
        code += "        }\n"

        # Get outputs
        code += "        auto s_loop_result = {cls_object_name}.process();\n".format(
            cls_object_name=cls_object_name
        )
        code += "        {condition_name} = std::get<0>(s_loop_result);\n".format(
            condition_name=condition_name
        )
        for i in range(self.node.carried_count):
            code += "        {carried_name} = std::move(std::get<{i}>(s_loop_result));\n".format(
                carried_name=self.fmt_dict["outputs"][i], i=i + 1
            )

        for i, scan_name in enumerate(scan_names):
            symbol = sub_graph_generator.graph.get_symbol(
                sub_graph_generator.graph.outputs[1 + self.node.carried_count + i]
            )
            if symbol.is_tensor():
                code += "        {scan_name}.push_back(std::move(std::get<{i}>(s_loop_result)));\n".format(
                    scan_name=scan_name, i=i + 1 + self.node.carried_count
                )
            elif symbol.is_vector():
                code += "        {scan_name}.insert({scan_name}.end(), std::get<{i}>(s_loop_result).begin(), std::get<{i}>(s_loop_result).end());\n".format(
                    scan_name=scan_name, i=i + 1 + self.node.carried_count
                )
            elif symbol.is_value():
                code += "        {scan_name}.push_back(std::get<{i}>(s_loop_result));\n".format(
                    scan_name=scan_name, i=i + 1 + self.node.carried_count
                )
            else:
                raise RuntimeError("Unknow error")

        if need_loop_count:
            code += "        {loop_count_name} = static_cast<unsigned>(i);\n".format(
                loop_count_name=loop_count_name
            )
        code += "    }\n"

        for i, scan_name in enumerate(scan_names):
            symbol = sub_graph_generator.graph.get_symbol(
                sub_graph_generator.graph.outputs[1 + self.node.carried_count + i]
            )
            output_name = str(self.fmt_dict["outputs"][i + self.node.carried_count])
            if symbol.is_tensor():
                code += "    auto {output_name} = tfcc::base::stack({scan_name}, 0);\n".format(
                    output_name=output_name, scan_name=scan_name
                )
            elif symbol.is_vector():
                code += "    auto {output_name} = tfcc::data::set({scan_name}, {{{loop_count_name}, static_cast<unsigned>({scan_name} / {loop_count_name})}});\n".format(
                    output_name=output_name,
                    scan_name=scan_name,
                    loop_count_name=loop_count_name,
                )
            elif symbol.is_value():
                code += "    auto {output_name} = std::move({scan_name});\n".format(
                    output_name=output_name, scan_name=scan_name
                )
            else:
                raise RuntimeError("Unknow error")
        return code
