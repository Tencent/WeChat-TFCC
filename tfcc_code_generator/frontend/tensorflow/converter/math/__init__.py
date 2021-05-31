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


def get_all_converters():
    from .abs import Abs
    from .argmax import ArgMax
    from .arithmetic import Add, Sub, Mul, Div, BiasAdd, AddV2, RealDiv
    from .batchmatmul import BatchMatMul
    from .batchmatmulv2 import BatchMatMulV2
    from .floordiv import FloorDiv
    from .leakyrelu import LeakyRelu
    from .matmul import MatMul
    from .maximum import Maximum
    from .minimum import Minimum
    from .pow import Pow
    from .reduce import (
        Mean,
        Sum,
        Prod,
        Max,
        Min,
        All,
        ReduceMean,
        ReduceSum,
        ReduceProd,
        ReduceMax,
        ReduceMin,
        ReduceAll,
    )
    from .relu import Relu
    from .rsqrt import Rsqrt
    from .reciprocal import Reciprocal
    from .sigmoid import Sigmoid
    from .sign import Sign
    from .sqrt import Sqrt
    from .square import Square
    from .squareddifference import SquaredDifference
    from .softmax import Softmax
    from .softplus import Softplus
    from .tanh import Tanh
    from .topkv2 import TopKV2

    return [
        Abs(),
        ArgMax(),
        Add(),
        Sub(),
        Mul(),
        Div(),
        RealDiv(),
        BiasAdd(),
        AddV2(),
        BatchMatMul(),
        BatchMatMulV2(),
        FloorDiv(),
        LeakyRelu(),
        MatMul(),
        Maximum(),
        Minimum(),
        Mean(),
        Pow(),
        ReduceMean(),
        Mean(),
        ReduceSum(),
        Sum(),
        ReduceProd(),
        Prod(),
        ReduceMax(),
        Max(),
        ReduceMin(),
        Min(),
        ReduceAll(),
        All(),
        Relu(),
        Rsqrt(),
        Reciprocal(),
        Sigmoid(),
        Sign(),
        Sqrt(),
        Square(),
        SquaredDifference(),
        Softmax(),
        Softplus(),
        Sum(),
        Tanh(),
        TopKV2(),
    ]
