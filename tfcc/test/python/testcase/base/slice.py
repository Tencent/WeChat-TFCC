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


class Slice(test.Test):
    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(1, 10, dims)

        a = np.random.uniform(-1.0, 1.0, shape)

        axis = np.random.randint(0, dims)
        start = np.random.randint(0, shape[axis])
        end = np.random.randint(start + 1, shape[axis] + 1)
        expect = np.split(a, [start, end], axis=axis)[1]
        return {
            "a": a,
            "axis": np.asarray([axis], dtype=np.int32),
            "start": np.asarray([start], dtype=np.int32),
            "end": np.asarray([end], dtype=np.int32),
            "expect": expect,
        }
