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


import numpy as np
import math
import test


class GeluAccurate(test.Test):
    @property
    def classPrefix(self):
        return "gelu_accurate"

    def erf(self, x):
        data = np.reshape(x, [-1]).tolist()
        data = [math.erf(v) for v in data]
        return np.reshape(np.asarray(data), x.shape)

    def calculate(self, x):
        return 0.5 * x * (1 + self.erf(x / math.sqrt(2)))

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)

        a = np.random.uniform(-1.0, 1.0, shape)

        expect = self.calculate(a)
        return {"a": a, "expect": expect}
