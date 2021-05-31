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


class MKLQuantizationMatmul(test.Test):
    @property
    def randomLoopCount(self):
        return 5

    def getQuantizedScaleInfo(self, minValue, maxValue):
        stepCnt = 1 << 8

        rangeAdjust = float(stepCnt) / (float(stepCnt) - 1.0)
        rg = (maxValue - minValue) * rangeAdjust
        rangeScale = float(stepCnt) / rg
        offset = int(round(minValue * rangeScale))

        return rangeScale, offset

    def getRandomTestDataWithSpecialShape(self, shapeA, shapeB):
        aMin = np.random.uniform(-5.0, 5.0)
        aMax = aMin + np.random.uniform(0.05, 5.0)
        _, aOffset = self.getQuantizedScaleInfo(aMin, aMax)

        bMin = np.random.uniform(-5.0, 5.0)
        bMax = bMin + np.random.uniform(0.05, 5.0)
        _, bOffset = self.getQuantizedScaleInfo(bMin, bMax)
        bOffset -= -128

        a = np.random.randint(0, 256, shapeA)
        b = np.random.randint(-128, 127, shapeB)

        expect = np.matmul((a + aOffset), (b + bOffset))
        return {
            "a": a,
            "b": b,
            "aMin": np.asarray([aMin], dtype=np.float32),
            "aMax": np.asarray([aMax], dtype=np.float32),
            "bMin": np.asarray([bMin], dtype=np.float32),
            "bMax": np.asarray([bMax], dtype=np.float32),
            "expect": expect,
        }

    def getRandomTestData(self):
        n = np.random.randint(1, 128)
        m = n * 6 + np.random.randint(1, 128)
        k = np.random.randint(1, 512)
        return self.getRandomTestDataWithSpecialShape([m, k], [k, n])

    def getRandomTestData2(self):
        m = np.random.randint(1, 512)
        n = int(m / 6) + np.random.randint(1, 128)
        k = np.random.randint(1, 512)
        return self.getRandomTestDataWithSpecialShape([m, k], [k, n])

    def getRandomTestData3(self):
        m = np.random.randint(1, 512)
        n = 1
        k = np.random.randint(1, 512)
        return self.getRandomTestDataWithSpecialShape([m, k], [k, n])
