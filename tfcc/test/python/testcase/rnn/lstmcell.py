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


class LSTMCell(test.Test):
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def getRandomTestData(self):
        batch = np.random.randint(1, 5)
        units = np.random.randint(64, 512)
        inputSize = np.random.randint(64, 512)

        inputs = np.random.uniform(-1.0, 1.0, (batch, inputSize)).astype(np.float32)
        kernel = np.random.uniform(-1.0, 1.0, (inputSize + units, units * 4)).astype(
            np.float32
        )
        bias = np.random.uniform(-1.0, 1.0, (units * 4,)).astype(np.float32)
        hState = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)
        csState = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)

        xh = np.dot(np.concatenate((inputs, hState), axis=1), kernel) + bias
        i, ci, f, o = np.split(xh, 4, axis=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f + 1)
        ci = np.tanh(ci)
        nextCs = ci * i + csState * f
        o = self.sigmoid(o)
        co = np.tanh(nextCs)
        expect = co * o

        return {
            "units": np.asarray([units], dtype=np.int32),
            "inputs": inputs,
            "h": hState,
            "cs": csState,
            "lstm_cell/kernel": kernel,
            "lstm_cell/bias": bias,
            "ecs": nextCs,
            "expect": expect,
        }
