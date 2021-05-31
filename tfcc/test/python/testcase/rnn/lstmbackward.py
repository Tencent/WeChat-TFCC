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
from numpy.lib.function_base import hamming
import test


class LSTMBackward(test.Test):
    @property
    def classPrefix(self):
        return "lstm_backward"

    @property
    def randomLoopCount(self):
        return 1

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def lstm(self, input, i_kernel, h_kernel, bias, h, c):
        xh = np.dot(input, i_kernel) + np.dot(h, h_kernel) + bias
        i, ci, f, o = np.split(xh, 4, axis=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        ci = np.tanh(ci)
        next_c = ci * i + c * f
        o = self.sigmoid(o)
        co = np.tanh(next_c)
        next_h = co * o
        return next_h, next_c

    def getRandomTestData(self):
        seq_length = np.random.randint(1, 8)
        batch = np.random.randint(1, 5)
        units = np.random.randint(32, 64)
        input_size = np.random.randint(32, 64)

        inputs = np.random.uniform(-1.0, 1.0, (seq_length, batch, input_size)).astype(
            np.float32
        )
        i_kernel = np.random.uniform(-1.0, 1.0, (input_size, units * 4)).astype(
            np.float32
        )
        h_kernel = np.random.uniform(-1.0, 1.0, (units, units * 4)).astype(np.float32)
        bias = np.random.uniform(-1.0, 1.0, (units * 4,)).astype(np.float32)
        ih = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)
        ic = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)

        h = ih
        c = ic
        outputs = []
        for i in range(seq_length):
            x = inputs[-i - 1]
            h, c = self.lstm(x, i_kernel, h_kernel, bias, h, c)
            outputs.append(h)

        expect = np.stack(outputs, axis=0)

        return {
            "inputs": inputs,
            "i_kernel": i_kernel,
            "h_kernel": h_kernel,
            "bias": bias,
            "ih": ih,
            "ic": ic,
            "expect": expect,
            "h": h,
            "c": c,
        }
