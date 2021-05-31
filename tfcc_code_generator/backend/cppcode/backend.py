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

import os
import typing
import argparse
import numpy as np
import ir.framework
from backend.cppcode.modelgenerator import ModelGenerator
from backend.cppcode.graphgenerator import GraphGenerator
from backend.cppcode.namemanager import NameManager


def ir2code(model: ir.framework.Model, entrance_cls_name: str):
    name_manager = NameManager()
    model_generator = ModelGenerator(model, entrance_cls_name, name_manager)
    return model_generator


def entrance(args, model: ir.framework.Model):
    class_name = args.classname
    if not class_name:
        class_name = model.graphs[model.entrance].name
    model_generator = ir2code(model, class_name)
    declaration, define, data = (
        model_generator.declaration,
        model_generator.define,
        model_generator.data,
    )
    declaration_header = ""
    declaration_header += "#pragma once\n\n"
    declaration_header += "#include <vector>\n"
    declaration_header += '#include "tfcc.h"\n'
    declaration_header += '#include "helper/tfcc_helper.h"\n'
    define_header = '#include "{prefix}.h"\n#include <cmath>\n'.format(
        prefix=args.file_prefix
    )
    if args.namespace:
        declaration = "{header}\nnamespace {namespace}\n{{\n{declaration}\n}}\n".format(
            header=declaration_header,
            namespace=args.namespace,
            declaration=declaration,
        )
        define = "{header}\nnamespace {namespace}\n{{\n{define}\n}}\n".format(
            header=define_header, namespace=args.namespace, define=define
        )
    else:
        declaration = "{header}\n{declaration}\n".format(
            header=declaration_header,
            declaration=declaration,
        )
        define = "{header}\n{define}\n".format(header=define_header, define=define)

    open(os.path.join(args.output_path, args.file_prefix + ".h"), "w").write(
        declaration
    )
    open(os.path.join(args.output_path, args.file_prefix + ".cpp"), "w").write(define)
    np.savez(os.path.join(args.output_path, args.file_prefix + ".npz"), **data)

    if not args.demo:
        return
    demo = ""
    demo += "#include <iostream>\n"
    demo += "#include <atomic>\n"
    demo += "#include <thread>\n"
    demo += '#include "tfcc_mkl.h"\n'
    demo += '#include "{prefix}.h"\n'.format(prefix=args.file_prefix)

    if args.namespace:
        demo += (
            "\nnamespace {namespace}\n{{\n{demo}\n}}\nusing {namespace}::Demo;".format(
                namespace=args.namespace,
                demo=model_generator.entrance_graph_generator.demo,
            )
        )
    else:
        demo += "\n" + model_generator.entrance_graph_generator.demo

    demo += """
int main(int argc, char* argv[])
{
    if (argc < 6)
    {
        std::cout << "Usage: " << argv[0] << " modeldata testdata devicecnt threadcnt logiccnt" << std::endl;
        return 1;
    }
    size_t deviceCnt = std::stoull(argv[3]);
    size_t threadCnt = std::stoull(argv[4]);
    size_t logicCnt = std::stoull(argv[5]);
    tfcc::initialize_mkl(deviceCnt, threadCnt);

    tfcc::MultiDataLoader loader;
    tfcc::NPZDataLoader modelLoader(argv[1]);
    tfcc::NPZDataLoader sampleLoader(argv[2]);
    loader.addLoader("model", modelLoader);
    loader.addLoader("sample", sampleLoader);
    tfcc::DataLoader::setGlobalDefault(&loader);

    Demo demo;
    demo.runOnce();

    Demo::Statistics statistics;

    for (size_t i = 0; i < logicCnt; ++i)
    {
        new std::thread(
            [deviceCnt, threadCnt, &statistics]()
            {
                tfcc::initialize_mkl(deviceCnt, threadCnt);
                Demo demo;
                demo.run(statistics);
            }
        );
    }

    std::cout << std::endl;
    while (true)
    {
        size_t oldCnt = statistics.processCnt.load();
        size_t oldCost = statistics.totalCost.load();
        std::this_thread::sleep_for(std::chrono::seconds(10));
        size_t newCnt = statistics.processCnt.load();
        size_t newCost = statistics.totalCost.load();

        std::cout << "qps: " << static_cast<double>(newCnt - oldCnt) / 10.
            << " avgcost: " << static_cast<double>(newCost - oldCost) / static_cast<double>(newCnt - oldCnt) / 1000 << std::endl;
    }
}
"""
    open(os.path.join(args.output_path, args.file_prefix + "_demo.cpp"), "w").write(
        demo
    )
    np.savez(
        os.path.join(args.output_path, args.file_prefix + "_testdata.npz"),
        **model_generator.entrance_graph_generator.demo_data
    )


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-path", required=True, help="Output base path")
    parser.add_argument("--file-prefix", required=True, help="Output file prefix")
    parser.add_argument("--namespace", default=None, help="Namespace of class")
    parser.add_argument("--classname", default=None, help="Name of class")
    parser.add_argument("--demo", action="store_true", help="Create demo file")
    return parser
