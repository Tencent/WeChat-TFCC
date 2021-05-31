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


import testcase.blas
import testcase.base
import testcase.math
import testcase.quantization
import testcase.rnn
import testcase.layer
import testcase.nn
import testcase.reduce
import testcase.relation
import testcase.signal

allModules = [
    testcase.blas,
    testcase.base,
    testcase.math,
    testcase.quantization,
    testcase.rnn,
    testcase.layer,
    testcase.nn,
    testcase.reduce,
    testcase.relation,
    testcase.signal,
]
