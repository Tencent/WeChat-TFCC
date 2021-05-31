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


class Equal(test.Test):
    def calculate(self, a, b):
        return np.asarray(a == b, dtype=np.uint8)

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)

        shapeAMask = np.random.randint(0, 2, dims)
        shapeA = [1 if shapeAMask[i] == 0 else shape[i] for i in range(dims)]
        shapeBMask = np.random.randint(0, 1, dims)
        shapeB = [1 if shapeBMask[i] == 0 else shape[i] for i in range(dims)]

        a1 = np.random.randint(0, 32, shapeA)
        b1 = np.random.randint(0, 32, shapeB)
        expect1 = self.calculate(a1, b1)

        a2 = np.reshape(a1, [-1]).tolist()[0]
        b2 = b1
        expect2 = self.calculate(a2, b2)

        a3 = a1
        b3 = np.reshape(b1, [-1]).tolist()[0]
        expect3 = self.calculate(a3, b3)

        return {
            "a1": a1,
            "b1": b1,
            "expect1": expect1,
            "a2": a2,
            "b2": b2,
            "expect2": expect2,
            "a3": a3,
            "b3": b3,
            "expect3": expect3,
        }
