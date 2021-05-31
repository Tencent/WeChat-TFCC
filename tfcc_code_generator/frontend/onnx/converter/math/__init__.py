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


def get_all_converters(op_set: dict):
    from .abs import Abs
    from .arithmetic import Add, Sub, Mul, Div
    from .clip import Clip
    from .einsum import Einsum
    from .erf import Erf
    from .leakyrelu import LeakyRelu
    from .gemm import Gemm
    from .matmul import MatMul
    from .pow import Pow
    from .prelu import PRelu
    from .reduce import ReduceMean, ReduceSum, ReduceProd
    from .relu import Relu
    from .sigmoid import Sigmoid
    from .softmax import Softmax
    from .sqrt import Sqrt
    from .tanh import Tanh
    from .logsoftmax import LogSoftmax
    from .softplus import Softplus
    from .max import Max
    from .min import Min
    from .log import Log
    from .neg import Neg
    from .reciprocal import Reciprocal
    from .topk import TopK
    from .sum import Sum

    return [
        Abs(op_set),
        Add(op_set),
        Sub(op_set),
        Mul(op_set),
        Div(op_set),
        Clip(op_set),
        Einsum(op_set),
        Erf(op_set),
        LeakyRelu(op_set),
        Gemm(op_set),
        MatMul(op_set),
        Pow(op_set),
        PRelu(op_set),
        ReduceMean(op_set),
        ReduceSum(op_set),
        ReduceProd(op_set),
        Relu(op_set),
        Sigmoid(op_set),
        Softmax(op_set),
        Sqrt(op_set),
        Tanh(op_set),
        LogSoftmax(op_set),
        Softplus(op_set),
        Max(op_set),
        Min(op_set),
        Log(op_set),
        Neg(op_set),
        Reciprocal(op_set),
        TopK(op_set),
        Sum(op_set),
    ]
