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
import io
import typing
import argparse
import zipfile
import numpy as np
import ir.framework
from backend.runtime.modelgenerator import ModelGenerator
from backend.runtime.namemanager import NameManager


def ir2proto(model: ir.framework.Model):
    name_manager = NameManager()
    model_generator = ModelGenerator(model, name_manager)
    model_pb, data_map = model_generator.process()

    f = io.BytesIO()
    np.savez(f, **data_map)

    return model_pb, f.getvalue()


def entrance(
    args, model: ir.framework.Model, model_with_fusion_ops: ir.framework.Model = None
):
    proto, data = ir2proto(model)
    zip_file = zipfile.ZipFile(args.output_path, "w")
    zip_file.writestr("model.pb", proto.SerializeToString())
    zip_file.writestr("model.npz", data)
    if model_with_fusion_ops is not None:
        proto, data = ir2proto(model_with_fusion_ops)
        zip_file.writestr("model.cpu.pb", proto.SerializeToString())
    if args.user_data_path:
        zip_file.write(args.user_data_path, arcname="userdata")


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-path", required=True, help="Output path")
    parser.add_argument(
        "--user-data-path",
        required=False,
        help="User data which could be get from runtime",
    )
    return parser
