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


class Add(test.Test):
    @property
    def randomLoopCount(self):
        return 20

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)

        shapeAMask = np.random.randint(0, 2, dims)
        shapeA = [1 if shapeAMask[i] == 0 else shape[i] for i in range(dims)]
        shapeBMask = np.random.randint(0, 1, dims)
        shapeB = [1 if shapeBMask[i] == 0 else shape[i] for i in range(dims)]

        a = np.random.uniform(-1.0, 1.0, shapeA)
        b = np.random.uniform(-1.0, 1.0, shapeB)

        expect = a + b
        return {"a": a, "b": b, "expect": expect}

    def getRandomTestData2(self):
        dims = np.random.randint(2, 5)
        shape = np.random.randint(1, 10, dims)
        cut = np.random.randint(1, dims)

        shapeAMask = np.random.randint(0, 2, dims)
        shapeA = [1 if shapeAMask[i] == 0 else shape[i] for i in range(dims)]
        shapeA = shapeA[cut:]
        shapeBMask = np.random.randint(0, 1, dims)
        shapeB = [1 if shapeBMask[i] == 0 else shape[i] for i in range(dims)]

        a = np.random.uniform(-1.0, 1.0, shapeA)
        b = np.random.uniform(-1.0, 1.0, shapeB)

        expect = a + b
        return {"a": a, "b": b, "expect": expect}

    def getRandomTestData3(self):
        dims = np.random.randint(2, 5)
        shape = np.random.randint(1, 10, dims)
        cut = np.random.randint(1, dims)

        shapeAMask = np.random.randint(0, 2, dims)
        shapeA = [1 if shapeAMask[i] == 0 else shape[i] for i in range(dims)]
        shapeBMask = np.random.randint(0, 1, dims)
        shapeB = [1 if shapeBMask[i] == 0 else shape[i] for i in range(dims)]
        shapeB = shapeB[cut:]

        a = np.random.uniform(-1.0, 1.0, shapeA)
        b = np.random.uniform(-1.0, 1.0, shapeB)

        expect = a + b
        return {"a": a, "b": b, "expect": expect}
