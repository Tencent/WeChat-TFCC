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

import ir
from ir.node import Node


def get_fusionop_maps():
    fusionop_map = {
        ir.node.math.Add: 0,
        ir.node.math.Sub: 1,
        ir.node.math.Mul: 2,
        ir.node.math.Div: 3,
        ir.node.math.Abs: 4,
        ir.node.math.Min: 5,
        ir.node.math.Max: 6,
        ir.node.math.Neg: 7,
        ir.node.math.Sqrt: 8,
        ir.node.math.Rsqrt: 9,
        ir.node.math.Relu: 10,
        ir.node.math.Tanh: 11,
        ir.node.math.LeakyRelu: 12,
        ir.node.math.Log: 13,
        ir.node.math.Sigmoid: 14,
        ir.node.math.Softplus: 15,
        ir.node.math.Reciprocal: 16,
        ir.node.math.Clip: 17,
    }
    return fusionop_map
