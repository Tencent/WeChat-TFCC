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


class UnsortedSegmentSum(test.Test):
    @property
    def classPrefix(self):
        return "unsorted_segment_sum"

    @property
    def randomLoopCount(self):
        return 20

    def getRandomTestData(self):
        dims = np.random.randint(1, 3)
        shape = np.random.randint(1, 32, dims)
        a = np.random.uniform(-1.0, 1.0, shape)

        num = np.random.randint(max(1, shape[0] - 5), shape[0] + 5)
        ids = np.random.randint(-1, num, shape[0])

        expect = np.zeros([num] + shape.tolist()[1:])
        for i, idx in enumerate(ids):
            if idx < 0:
                continue
            expect[idx] += a[i]

        return {
            "a": a,
            "ids": ids,
            "num": np.asarray([num], dtype=np.int32),
            "expect": expect,
        }
