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


class PyTorchLSTMCell(test.Test):
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def getRandomTestData(self):
        batch = np.random.randint(1, 5)
        units = np.random.randint(64, 512)
        inputSize = np.random.randint(64, 512)

        inputs = np.random.uniform(-1.0, 1.0, (batch, inputSize)).astype(np.float32)
        i_kernel = np.random.uniform(-1.0, 1.0, (inputSize, units * 4)).astype(
            np.float32
        )
        i_bias = np.random.uniform(-1.0, 1.0, (units * 4,)).astype(np.float32)
        h_kernel = np.random.uniform(-1.0, 1.0, (units, units * 4)).astype(np.float32)
        h_bias = np.random.uniform(-1.0, 1.0, (units * 4,)).astype(np.float32)
        h_state = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)
        cs_state = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)

        xh = np.dot(inputs, i_kernel) + i_bias + np.dot(h_state, h_kernel) + h_bias
        i, f, ci, o = np.split(xh, 4, axis=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        ci = np.tanh(ci)
        next_cs = ci * i + cs_state * f
        o = self.sigmoid(o)
        co = np.tanh(next_cs)
        expect = co * o

        return {
            "units": np.asarray([units], dtype=np.int32),
            "inputs": inputs,
            "h": h_state,
            "cs": cs_state,
            "lstm_cell/weight_ih": i_kernel,
            "lstm_cell/bias_ih": i_bias,
            "lstm_cell/weight_hh": h_kernel,
            "lstm_cell/bias_hh": h_bias,
            "ecs": next_cs,
            "expect": expect,
        }
