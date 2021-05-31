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


import math
import numpy as np
import test


class LocalResponseNormalization(test.Test):
    @property
    def classPrefix(self):
        return "local_response_normalization"

    @property
    def randomLoopCount(self):
        return 20

    def getRandomTestData(self):
        paramsDims = np.random.randint(1, 5)
        axis = np.random.randint(0, paramsDims)
        paramsShape = np.random.randint(1, 10, paramsDims)

        n = int(np.prod(paramsShape[:axis]))
        c = paramsShape[axis]
        d = int(np.prod(paramsShape[axis + 1 :]))

        params = np.random.uniform(-1.0, 1.0, [c, n, d])

        alpha = np.random.uniform(0.001, 0.0001)
        beta = np.random.uniform(0.5, 0.99)
        bias = np.random.uniform(1.0, 2.0)
        size = np.random.randint(1, c + 5)

        expect = []
        for i, x in enumerate(params):
            start = max(0, i - math.floor((size - 1) / 2))
            end = min(i - 1, i + math.ceil((size - 1) / 2)) + 1
            squareSum = np.sum(params[start:end] * params[start:end], axis=0)
            y = x / (bias + alpha / size * squareSum) ** beta
            expect.append(y)

        expect = np.asarray(expect)

        params = np.transpose(params, [1, 0, 2])
        params = np.reshape(params, paramsShape)
        expect = np.transpose(expect, [1, 0, 2])
        expect = np.reshape(expect, paramsShape)

        return {
            "a": params,
            "axis": axis,
            "alpha": alpha,
            "beta": beta,
            "bias": bias,
            "size": size,
            "expect": expect,
        }
