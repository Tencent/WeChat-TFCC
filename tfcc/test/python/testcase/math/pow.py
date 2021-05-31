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
import test


class Pow(test.Test):
    def calculate(self, x, exponent):
        return np.power(x, exponent)

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)

        a = np.random.uniform(0.0, 1.0, shape)
        exponent = np.random.uniform(-5.0, 5.0)

        expect = self.calculate(a, exponent)
        return {"a": a, "exponent": np.asarray([exponent]), "expect": expect}
