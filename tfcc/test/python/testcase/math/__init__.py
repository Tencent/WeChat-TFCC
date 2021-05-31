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


from .add import Add
from .sub import Sub
from .mul import Mul
from .div import Div
from .batch_add import BatchAdd
from .batch_mul import BatchMul
from .transform import Transform
from .transform2 import Transform2
from .transform3 import Transform3
from .transform4 import Transform4
from .transform5 import Transform5
from .transform6 import Transform6
from .min import Min
from .max import Max
from .sigmoid import Sigmoid
from .relu import Relu
from .leaky_relu import LeakyRelu
from .softplus import Softplus
from .log import Log
from .rsqrt import Rsqrt
from .tanh import Tanh
from .sin import Sin
from .cos import Cos
from .pow import Pow
from .pow_v2 import PowV2
from .pow_v3 import PowV3
from .softmax import Softmax
from .gelu import Gelu
from .gelu_accurate import GeluAccurate
from .erf import Erf
from .unsorted_segment_sum import UnsortedSegmentSum
from .abs import Abs
from .argmax import ArgMax
from .asin import Asin
from .asinh import Asinh
from .acos import Acos
from .acosh import Acosh
from .atan import Atan
from .atanh import Atanh
from .sign import Sign

allTests = [
    Add,
    Sub,
    Mul,
    Div,
    BatchAdd,
    BatchMul,
    Transform,
    Transform2,
    Transform3,
    Transform4,
    Transform5,
    Transform6,
    Min,
    Max,
    Sigmoid,
    Relu,
    LeakyRelu,
    Softplus,
    Log,
    Rsqrt,
    Tanh,
    Sin,
    Cos,
    Pow,
    PowV2,
    PowV3,
    Softmax,
    Gelu,
    GeluAccurate,
    Erf,
    UnsortedSegmentSum,
    Abs,
    ArgMax,
    Asin,
    Asinh,
    Acos,
    Acosh,
    Atan,
    Atanh,
    Sign,
]
