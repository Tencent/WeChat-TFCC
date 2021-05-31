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


class RFFT(test.Test):
    @property
    def classPrefix(self):
        return "rfft"

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)

        a = np.random.uniform(-10, 10, shape)
        length = a.shape[-1]

        expect = np.fft.rfft(a, length)
        expect = np.stack([expect.real, expect.imag], axis=-1)

        return {
            "a": a,
            "length": length,
            "expect": expect,
        }

    def getRandomTestData2(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)

        a = np.random.uniform(-10, 10, shape)
        length = np.random.randint(1, shape[-1])

        expect = np.fft.rfft(a, length)
        expect = np.stack([expect.real, expect.imag], axis=-1)

        return {
            "a": a,
            "length": length,
            "expect": expect,
        }

    def getRandomTestData3(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)

        a = np.random.uniform(-10, 10, shape)
        length = np.random.randint(shape[-1], shape[-1] + 5)

        expect = np.fft.rfft(a, length)
        expect = np.stack([expect.real, expect.imag], axis=-1)

        return {
            "a": a,
            "length": length,
            "expect": expect,
        }
