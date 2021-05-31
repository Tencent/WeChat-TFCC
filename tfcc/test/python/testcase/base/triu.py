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


class Triu(test.Test):
    @property
    def randomLoopCount(self):
        return 20

    def getRandomTestData(self):
        k = np.random.randint(-10, 10)
        shape = np.random.randint(1, 10, 2)

        a = np.random.uniform(-1.0, 1.0, shape)

        expect = np.triu(a, k)

        return {"a": a, "k": np.asarray([k], dtype=np.int64), "expect": expect}
