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

import logging
import numpy as np
import ir.framework
from backend.cppcode.namemanager import NameManager
from backend.cppcode.common import get_symbol_cpp_dtype, symbol_to_cpp_type
from backend.cppcode.nodegenerator import get_all_node_generators


class GraphGenerator(object):
    def __init__(self, graph: ir.framework.Graph, cls_name: str, model_generator):
        self._graph = graph
        self._cls_name = model_generator.name_manager.get_symbol_name(graph, cls_name)
        self._scope_name = self._cls_name
        self._model_generator = model_generator

    @property
    def graph(self):
        return self._graph

    @property
    def cls_name(self):
        return self._cls_name

    @property
    def model_generator(self):
        return self._model_generator

    @property
    def name_manager(self):
        return self.model_generator.name_manager

    @property
    def input_symbol_names(self):
        names = []
        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            symbol_name = self.name_manager.get_symbol_name(symbol, symbol.name)
            names.append(symbol_name)
        return names

    @property
    def declaration(self):
        symbol_declarations = ""
        for symbol in sorted(self.graph.symbols.values(), key=lambda a: a.name):
            if symbol.name not in self.graph.keep_symbol_names:
                continue
            assert symbol.is_constant() and symbol.stype == symbol.origin_stype
            symbol_name = self.name_manager.get_symbol_name(symbol)
            symbol_declarations += "    {symbol} {name};\n".format(
                symbol=symbol_to_cpp_type(symbol, False), name=symbol_name
            )

        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            symbol_name = self.name_manager.get_symbol_name(symbol, symbol.name)
            symbol_declarations += "    {symbol} {name};\n".format(
                symbol=symbol_to_cpp_type(symbol, False), name=symbol_name
            )

        function_declarations = "    {cls_name}();\n".format(cls_name=self._cls_name)
        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            symbol_name = self.name_manager.get_symbol_name(symbol, symbol.name)
            function_declarations += "    /**\n"
            function_declarations += (
                "     * @param v_{name} {origin_name} {shape}\n".format(
                    name=symbol_name, origin_name=symbol.name, shape=symbol.shape
                )
            )
            function_declarations += "     */\n"
            function_declarations += (
                "    void set_{name}(const {symbol}& v_{name});\n".format(
                    name=symbol_name, symbol=symbol_to_cpp_type(symbol, True)
                )
            )

        return_types = []
        for name in self.graph.outputs:
            symbol = self.graph.get_symbol(name)
            return_types.append(symbol_to_cpp_type(symbol, False, True))

        processFunctionDeclaration = ""
        processFunctionDeclaration += "    /**\n"
        processFunctionDeclaration += "     * @return\n"
        for i, name in enumerate(self.graph.outputs):
            symbol = self.graph.get_symbol(name)
            processFunctionDeclaration += "     * output:{i} {name} {shape}\n".format(
                i=i, name=symbol.name, shape=symbol.shape
            )
        processFunctionDeclaration += "     */\n"
        processFunctionDeclaration += "    std::tuple<{types}> process();\n".format(
            types=", ".join(return_types)
        )
        format_str = """
class {cls_name}
{{
{symbols}
public:
{functions}
{process}
}};
"""
        return format_str.format(
            cls_name=self._cls_name,
            symbols=symbol_declarations,
            functions=function_declarations,
            process=processFunctionDeclaration,
        )

    @property
    def define(self):
        construct_function = "{cls_name}::{cls_name}()\n".format(
            cls_name=self._cls_name
        )
        construct_function += "{\n"
        construct_function += (
            '    auto scopeG = tfcc::Scope::scope("{scope}");\n'.format(
                scope=self._scope_name
            )
        )
        for symbol in sorted(self.graph.symbols.values(), key=lambda a: a.name):
            if symbol.name not in self.graph.keep_symbol_names:
                continue
            assert symbol.is_constant() and symbol.stype == symbol.origin_stype
            symbol_name = self.name_manager.get_symbol_name(symbol)
            if symbol.is_tensor():
                construct_function += '    {name} = tfcc::View<{dtype}>(tfcc::Constant<{dtype}>::getConstant("{name}"));\n'.format(
                    name=symbol_name, dtype=get_symbol_cpp_dtype(symbol)
                )
            elif symbol.is_value() and symbol.dtype != ir.framework.DataType.BOOL:
                construct_function += '    {name} = tfcc::Configure<{dtype}>::getConfigure("{name}");\n'.format(
                    name=symbol_name, dtype=get_symbol_cpp_dtype(symbol)
                )
            elif symbol.is_value() and symbol.dtype == ir.framework.DataType.BOOL:
                construct_function += '    {name} = tfcc::Configure<uint8_t>::getConfigure("{name}");\n'.format(
                    name=symbol_name, dtype=get_symbol_cpp_dtype(symbol)
                )
            elif symbol.is_vector():
                assert symbol.dtype != ir.framework.DataType.BOOL
                construct_function += '    {name} = tfcc::data::get(tfcc::Constant<{dtype}>::getConstant("{name}"));\n'.format(
                    name=symbol_name, dtype=get_symbol_cpp_dtype(symbol)
                )
            else:
                raise RuntimeError("Unknow constant symbol")
        construct_function += "}\n"

        setter_function = ""
        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            symbol_name = self.name_manager.get_symbol_name(symbol, symbol.name)
            setter_function += (
                "void {cls_name}::set_{name}(const {symbol}& v_{name})\n".format(
                    cls_name=self._cls_name,
                    name=symbol_name,
                    symbol=symbol_to_cpp_type(symbol, True),
                )
            )
            setter_function += "{\n"
            if symbol.is_tensor():
                setter_function += (
                    "    {name} = tfcc::View<{dtype}>(v_{name});\n".format(
                        name=symbol_name, dtype=get_symbol_cpp_dtype(symbol)
                    )
                )
            elif symbol.is_value():
                setter_function += "    {name} = v_{name};\n".format(name=symbol_name)
            elif symbol.is_vector():
                setter_function += "    {name} = v_{name};\n".format(name=symbol_name)
            else:
                raise RuntimeError("Unknow stype")
            setter_function += "}\n"

        return_types = []
        for name in self.graph.outputs:
            symbol = self.graph.get_symbol(name)
            return_types.append(symbol_to_cpp_type(symbol, False, True))

        generator_classes = get_all_node_generators()
        process_function = "std::tuple<{types}> {cls_name}::process()\n".format(
            cls_name=self._cls_name, types=", ".join(return_types)
        )
        process_function += "{\n"
        all_succ = True
        for node in self.graph.nodes:
            generator = None
            for generator_class in generator_classes:
                if generator_class.accept(node):
                    generator = generator_class(node, self)
            if generator:
                process_function += generator.comment
                process_function += "    " + generator.code
                # process_function += generator.debug_code
            else:
                all_succ = False
                logging.debug(node.__class__)
        if not all_succ:
            raise RuntimeError("Node to code error")
        return_codes = []
        for name in self.graph.outputs:
            if (
                self.graph.get_symbol(name).is_tensor()
                and self.graph.get_symbol(name).stype
                != ir.framework.SymbolType.VARIABLE
            ):
                return_codes.append(
                    "tfcc::data::copy({symbol})".format(
                        symbol=self.name_manager.get_symbol_name(
                            self.graph.get_symbol(name)
                        )
                    )
                )
            else:
                return_codes.append(
                    "std::move({symbol})".format(
                        symbol=self.name_manager.get_symbol_name(
                            self.graph.get_symbol(name)
                        )
                    )
                )

        outputs_vals = ", ".join(return_codes)
        process_function += "    return std::make_tuple({vals});\n".format(
            vals=outputs_vals
        )
        process_function += "}\n"
        format_str = """
{constructer}

{setter}

{processer}

"""
        return format_str.format(
            cls_name=self._cls_name,
            constructer=construct_function,
            setter=setter_function,
            processer=process_function,
        )

    @property
    def data(self):
        symbols = {}
        for symbol in sorted(self.graph.symbols.values(), key=lambda a: a.name):
            if not symbol.is_constant():
                continue
            symbol_name = self.name_manager.get_symbol_name(symbol)
            key = "{scope}/{name}".format(scope=self._scope_name, name=symbol_name)
            if symbol.dtype == ir.framework.DataType.BOOL:
                symbols[key] = symbol.data.astype(np.uint8)
            else:
                symbols[key] = symbol.data
        return symbols

    @property
    def demo(self):
        constructer_body = ""
        constructer_body += "        {\n"
        constructer_body += '            auto scopeG = tfcc::Scope::scope("model");\n'
        constructer_body += "            _model = std::unique_ptr<{cls_name}>(new {cls_name});\n".format(
            cls_name=self._cls_name
        )
        constructer_body += "        }\n"
        constructer_body += "        {\n"
        constructer_body += '            auto scopeG = tfcc::Scope::scope("sample");\n'
        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            symbol_name = self.name_manager.get_symbol_name(symbol)
            fmtDict = {
                "name": name,
                "symbol_name": symbol_name,
                "dtype": get_symbol_cpp_dtype(symbol),
            }
            if symbol.is_tensor():
                constructer_body += '            _model->set_{symbol_name}(tfcc::Constant<{dtype}>::getConstant("{name}"));\n'.format(
                    **fmtDict
                )
            elif symbol.is_value() and symbol.dtype != ir.framework.DataType.BOOL:
                constructer_body += '            _model->set_{symbol_name}(tfcc::Configure<{dtype}>::getConfigure("{name}"));\n'.format(
                    **fmtDict
                )
            elif symbol.is_value() and symbol.dtype == ir.framework.DataType.BOOL:
                constructer_body += '            _model->set_{symbol_name}(tfcc::Configure<uint8_t>::getConfigure("{name}"));\n'.format(
                    **fmtDict
                )
            elif symbol.is_vector():
                assert symbol.dtype != ir.framework.DataType.BOOL
                constructer_body += '            _model->set_{symbol_name}(tfcc::data::get(tfcc::Constant<{dtype}>::getConstant("{name}")));\n'.format(
                    **fmtDict
                )
            else:
                raise RuntimeError("Unknow constant symbol")
        constructer_body += "        }\n"

        run_once_body = ""
        run_once_body += "        _model->process();\n"
        run_once_body += "        tfcc::MKLDevice* device = static_cast<tfcc::MKLDevice*>(tfcc::Device::getThreadDefault());\n"
        run_once_body += "        device->clearStatistics();\n"
        run_once_body += "        tfcc::Session::getThreadDefault()->sync();\n"
        run_once_body += "        auto result = _model->process();\n"
        run_once_body += "        tfcc::Session::getThreadDefault()->sync();\n"
        run_once_body += "        device->getStatistics().print();\n"
        for i, name in enumerate(self.graph.outputs):
            symbol = self.graph.get_symbol(name)
            if symbol.is_tensor() or symbol.is_value():
                run_once_body += '        std::cout << "{name}: " << std::get<{i}>(result) << std::endl;\n'.format(
                    name=name, i=i
                )
            elif symbol.is_vector():
                assert symbol.dtype != ir.framework.DataType.BOOL
                run_once_body += '        std::cout << "{name}: " << tfcc::data::set(std::get<{i}>(result), {static_cast<unsigned>(std::get<{i}>(result).size())}) << std::endl;\n'.format(
                    name=name, i=i
                )
            else:
                raise RuntimeError("Unknow constant symbol")

        code_fmt = """
class Demo
{{
    std::unique_ptr<{cls_name}> _model;

public:
    struct Statistics
    {{
        std::atomic<size_t> processCnt;
        std::atomic<size_t> totalCost;
    }};

public:
    Demo()
    {{
{constructer_body}
    }}

    void runOnce()
    {{
        auto scopeG = tfcc::Scope::scope("model");
{run_once_body}
    }}

    void run(Statistics& statistics)
    {{
        auto scopeG = tfcc::Scope::scope("model");
        while (true)
        {{
            tfcc::Coster coster;
            _model->process();
            tfcc::Session::getThreadDefault()->sync();
            ++statistics.processCnt;
            statistics.totalCost += coster.lap().microseconds();
        }}
    }}
}};
"""
        return code_fmt.format(
            cls_name=self._cls_name,
            constructer_body=constructer_body,
            run_once_body=run_once_body,
        )

    @property
    def demo_data(self):
        data = {}
        for name in self.graph.inputs:
            symbol = self.graph.get_symbol(name)
            shape = [1 if isinstance(s, str) else s for s in symbol.shape]
            if symbol.is_integer():
                data[name] = np.ones(shape)
            else:
                data[name] = np.zeros(shape) + 0.1
        return data
