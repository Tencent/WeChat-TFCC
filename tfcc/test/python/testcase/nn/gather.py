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


class Gather(test.Test):
    def getRandomTestData(self):
        paramsDims = np.random.randint(1, 5)
        paramsShape = np.random.randint(1, 10, paramsDims)

        params = np.random.uniform(-1.0, 1.0, paramsShape)

        indicesDims = np.random.randint(1, 5)
        indicesShape = np.random.randint(1, 10, indicesDims)

        indices = np.random.randint(0, paramsShape[0], indicesShape)

        expect = np.take(params, indices, axis=0)
        return {"params": params, "indices": indices, "expect": expect}
