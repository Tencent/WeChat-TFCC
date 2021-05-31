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


class Quantize(test.Test):
    @property
    def randomLoopCount(self):
        return 20

    def quantizeOnce(self, a, scale, offset, minValue, maxValue):
        quantized = int(round(a * scale)) - offset
        quantized += minValue
        quantized = max(quantized, minValue)
        quantized = min(quantized, maxValue)
        return quantized

    def getQuantizedScaleInfo(self, rangeMin, rangeMax, bits):
        stepCnt = 1 << bits
        rangeAdjust = stepCnt / (stepCnt - 1.0)
        r = (rangeMax - rangeMin) * rangeAdjust
        scale = stepCnt / r
        offset = int(round(rangeMin * scale))
        return scale, offset

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)

        a = np.random.uniform(-5.0, 5.0, shape).astype(np.float32)
        scale, offset = self.getQuantizedScaleInfo(np.min(a), np.max(a), 8)

        int8_a = [self.quantizeOnce(v, scale, offset, -128, 127) for v in a.flat]
        uint8_a = [self.quantizeOnce(v, scale, offset, 0, 255) for v in a.flat]
        int8_a = np.asarray(int8_a, dtype=np.int8)
        uint8_a = np.asarray(uint8_a, dtype=np.uint8)
        int8_a = int8_a.reshape(a.shape)
        uint8_a = uint8_a.reshape(a.shape)
        return {"a": a, "int8_a": int8_a, "uint8_a": uint8_a}
