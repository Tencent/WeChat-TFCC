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


class IRFFT(test.Test):
    @property
    def classPrefix(self):
        return "irfft"

    def getRandomTestData(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)
        shape = [2]

        a = np.zeros(shape, dtype=np.complex)
        a.real = np.random.uniform(-10, 10, shape)
        a.imag = np.random.uniform(-10, 10, shape)
        length = (a.shape[-1] - 1) * 2

        expect = np.fft.irfft(a, length)

        return {
            "a": np.stack([a.real, a.imag], axis=-1),
            "length": length,
            "expect": expect,
        }

    def getRandomTestData2(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)
        shape = [2]

        a = np.zeros(shape, dtype=np.complex)
        a.real = np.random.uniform(-10, 10, shape)
        a.imag = np.random.uniform(-10, 10, shape)
        length = np.random.randint(1, (a.shape[-1] - 1) * 2)

        expect = np.fft.irfft(a, length)

        return {
            "a": np.stack([a.real, a.imag], axis=-1),
            "length": length,
            "expect": expect,
        }

    def getRandomTestData3(self):
        dims = np.random.randint(1, 5)
        shape = np.random.randint(2, 10, dims)
        shape = [2]

        a = np.zeros(shape, dtype=np.complex)
        a.real = np.random.uniform(-10, 10, shape)
        a.imag = np.random.uniform(-10, 10, shape)
        length = np.random.randint((a.shape[-1] - 1) * 2, (a.shape[-1] - 1) * 2 + 5)

        expect = np.fft.irfft(a, length)

        return {
            "a": np.stack([a.real, a.imag], axis=-1),
            "length": length,
            "expect": expect,
        }
