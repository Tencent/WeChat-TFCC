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


class Matmul(test.Test):
    @property
    def randomLoopCount(self):
        return 5

    def getRandomTestData(self):
        m, n, k = np.random.randint(1, 100, 3)
        a = np.random.uniform(-1.0, 1.0, m * k)
        b = np.random.uniform(-1.0, 1.0, n * k)

        a = np.asarray(a, dtype=np.float32).reshape([m, k])
        b = np.asarray(b, dtype=np.float32).reshape([k, n])
        expect = np.dot(a, b)
        return {"a": a, "b": b, "expect": expect}

    def getRandomTestData2(self):
        m, n, k = np.random.randint(1, 100, 3)
        bs = np.random.randint(1, 6)
        a = np.random.uniform(-1.0, 1.0, bs * m * k)
        b = np.random.uniform(-1.0, 1.0, n * k)

        a = np.asarray(a, dtype=np.float32).reshape([bs, m, k])
        b = np.asarray(b, dtype=np.float32).reshape([k, n])
        expect = np.dot(a, b)
        return {"a": a, "b": b, "expect": expect}

    def getRandomTestData3(self):
        m, n, k = np.random.randint(1, 100, 3)
        bs = np.random.randint(1, 6)
        a = np.random.uniform(-1.0, 1.0, m * k)
        b = np.random.uniform(-1.0, 1.0, bs * n * k)

        a = np.asarray(a, dtype=np.float32).reshape([m, k])
        b = np.asarray(b, dtype=np.float32).reshape([bs, k, n])
        expect = np.dot(a, b).transpose([1, 0, 2])
        return {"a": a, "b": b, "expect": expect}

    def getRandomTestData4(self):
        m, n, k = np.random.randint(1, 100, 3)
        bs = np.random.randint(1, 6)
        a = np.random.uniform(-1.0, 1.0, bs * m * k)
        b = np.random.uniform(-1.0, 1.0, bs * n * k)

        a = np.asarray(a, dtype=np.float32).reshape([bs, m, k])
        b = np.asarray(b, dtype=np.float32).reshape([bs, k, n])
        cs = []
        for i in range(bs):
            cs.append(np.dot(a[i], b[i]))
        expect = np.stack(cs, axis=0)
        return {"a": a, "b": b, "expect": expect}
