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


class ArgMax(test.Test):
    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)
        axis = np.random.randint(0, dims)

        a = np.random.uniform(-10, 10, shape)

        expect = np.argmax(a, axis)
        expect = np.expand_dims(expect, axis)

        return {"a": a, "axis": axis, "expect": expect}

    def getRandomTestData2(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)
        axis = dims - 1

        a = np.random.uniform(-10, 10, shape)

        expect = np.argmax(a, axis)
        expect = np.expand_dims(expect, axis)

        return {"a": a, "axis": axis, "expect": expect}
