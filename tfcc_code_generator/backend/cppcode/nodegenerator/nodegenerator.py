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
import ir.framework
from backend.cppcode.namemanager import NameManager
from backend.cppcode.common import get_symbol_cpp_dtype


class SymbolWrapper(object):
    def __init__(self, symbol: ir.framework.Symbol, name_manager: NameManager):
        self._symbol = symbol
        self._name_manager = name_manager

    def __str__(self):
        return self._name_manager.get_symbol_name(self._symbol)

    @property
    def dtype(self):
        return get_symbol_cpp_dtype(self._symbol)


class NodeGenerator(object):
    def __init__(self, node: ir.node.Node, graph_generator):
        self._node = node
        self._graph_generator = graph_generator

    @classmethod
    def accept(cls, node: ir.node.Node):
        return False

    @property
    def graph_generator(self):
        return self._graph_generator

    @property
    def model_generator(self):
        return self.graph_generator.model_generator

    @property
    def name_manager(self) -> NameManager:
        return self.graph_generator.name_manager

    @property
    def node(self):
        return self._node

    @property
    def inputs(self):
        return self.node.inputs

    @property
    def outputs(self):
        return self.node.outputs

    @property
    def code(self):
        raise NotImplementedError

    @property
    def comment(self):
        code = ""
        code += "    /**\n"
        code += "     * node: {node}\n".format(node=self.node.__class__)
        code += "     * name: {name}\n".format(name=self.node.name)
        for i, inp in enumerate(self.inputs):
            code += "     * input{i}: {dtype}/{stype} {shape} {name}\n".format(
                i=i,
                dtype=inp.dtype,
                stype=inp.origin_stype if inp.origin_stype else inp.stype,
                shape=inp.shape,
                name=inp.name,
            )
        for i, out in enumerate(self.outputs):
            if isinstance(out.incomplete_data, list):
                code += "     * output{i}: {dtype}/{stype} {shape} {name} incomplete_data: {incomplete_data}\n".format(
                    i=i,
                    dtype=out.dtype,
                    stype=out.origin_stype if out.origin_stype else out.stype,
                    shape=out.shape,
                    name=out.name,
                    incomplete_data=out.incomplete_data,
                )
            else:
                code += "     * output{i}: {dtype}/{stype} {shape} {name}\n".format(
                    i=i,
                    dtype=out.dtype,
                    stype=out.origin_stype if out.origin_stype else out.stype,
                    shape=out.shape,
                    name=out.name,
                )
        code += "     */\n"
        return code

    @property
    def debug_code(self):
        code = ""
        for out in self.outputs:
            if out.is_vector():
                code += '    std::cout << "{name}: " << tfcc::data::set({output}, {{static_cast<unsigned>({output}.size())}}) << std::endl;\n'.format(
                    name=out.name, output=SymbolWrapper(out, self.name_manager)
                )
            else:
                code += (
                    '    std::cout << "{name}: " << {output} << std::endl;\n'.format(
                        name=out.name, output=SymbolWrapper(out, self.name_manager)
                    )
                )
        return code

    @property
    def fmt_dict(self):
        return {
            "inputs": [
                SymbolWrapper(symbol, self.name_manager) for symbol in self._node.inputs
            ],
            "outputs": [
                SymbolWrapper(symbol, self.name_manager)
                for symbol in self._node.outputs
            ],
            "node": self._node,
        }
