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

import sys
import json
import time
import logging
import argparse
import ir


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--frontend",
        choices=["onnx", "tensorflow"],
        required=True,
        help="Frontend type",
    )
    parser.add_argument(
        "--backend", choices=["cpp", "runtime"], required=True, help="Backend type"
    )
    parser.add_argument(
        "--enable-fusionop",
        action="store_true",
        help="Enable fusionop to improving inference",
    )
    parser.add_argument("--summary", help="Model summary output path")
    parser.add_argument(
        "--exclude-optimizers",
        nargs="+",
        required=False,
        help="Disable some optimizers",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug log")
    parser.add_argument(
        "--disable-optimization", action="store_true", help="Disable all optimization"
    )
    return parser


def main():
    parser = get_args_parser()
    args = parser.parse_known_args()[0]

    log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level, format="%(asctime)s %(levelname)s: %(message)s"
    )

    parsers = [parser]

    if args.frontend == "onnx":
        import frontend.onnx.frontend

        parsers.append(frontend.onnx.frontend.get_args_parser())
        frontend_entrance = frontend.onnx.frontend.entrance
    elif args.frontend == "tensorflow":
        import frontend.tensorflow.frontend

        parsers.append(frontend.tensorflow.frontend.get_args_parser())
        frontend_entrance = frontend.tensorflow.frontend.entrance
    else:
        raise RuntimeError("Unknow frontend type: {}".format(args.frontend))

    if args.backend == "cpp":
        import backend.cppcode.backend

        parsers.append(backend.cppcode.backend.get_args_parser())
        backend_entrance = backend.cppcode.backend.entrance
    elif args.backend == "runtime":
        import backend.runtime.backend

        parsers.append(backend.runtime.backend.get_args_parser())
        backend_entrance = backend.runtime.backend.entrance
    else:
        raise RuntimeError("Unknow backend type: {}".format(args.frontend))

    parser = argparse.ArgumentParser(parents=parsers)
    args = parser.parse_args()

    ts = time.time()
    model = frontend_entrance(args)
    if not model.verify():
        raise RuntimeError("Model verify failed")
    logging.info("Frontend cost {:.4} s".format(time.time() - ts))

    logging.info("Model summary before optimize: {}".format(model.summary))

    model_with_fusion_ops = None
    if not args.disable_optimization:
        ts = time.time()
        model = ir.optimize(model, args.exclude_optimizers)
        logging.info("Optimize cost {:.4} s".format(time.time() - ts))
        logging.info("Model summary after optimize: {}".format(model.summary))
        if args.backend == "runtime" and args.enable_fusionop:
            model_with_fusion_ops = ir.fusion(model)

    ts = time.time()
    if args.backend == "runtime":
        backend_entrance(args, model, model_with_fusion_ops)
    elif args.backend == "cpp":
        backend_entrance(args, model)

    logging.info("Backend cost {:.4} s".format(time.time() - ts))

    if args.summary is not None:
        summary = {
            "inputs": [],
            "outputs": [],
            "graphs": [],
            "entrance": model.entrance,
        }
        for graph in model.graphs.values():
            summary["graphs"].append(
                {
                    "name": graph.name,
                    "node_count": len(graph.nodes),
                    "constant_count": len(graph.keep_symbol_names),
                }
            )
        for name in model.graphs[model.entrance].inputs:
            symbol = model.graphs[model.entrance].get_symbol(name)
            summary["inputs"].append(
                {
                    "name": symbol.name,
                    "dtype": str(symbol.dtype),
                    "stype": str(symbol.stype),
                    "shape": symbol.shape,
                }
            )

        for name in model.graphs[model.entrance].outputs:
            symbol = model.graphs[model.entrance].get_symbol(name)
            summary["outputs"].append(
                {
                    "name": symbol.name,
                    "dtype": str(symbol.dtype),
                    "stype": str(symbol.stype),
                    "shape": symbol.shape,
                }
            )

        str_json = json.dumps(summary, indent=4)
        if args.summary == "-":
            print(str_json)
        else:
            open(args.summary, "w").write(str_json)


if __name__ == "__main__":
    main()
