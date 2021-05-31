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


class GRUCell(test.Test):
    def sigmoid(self, value):
        return 1 / (1 + np.exp(-value))

    def getRandomTestData(self):
        batch = np.random.randint(1, 5)
        units = np.random.randint(64, 512)
        input_size = np.random.randint(64, 512)

        inputs = np.random.uniform(-1.0, 1.0, (batch, input_size)).astype(np.float32)
        g_kernel = np.random.uniform(-1.0, 1.0, (input_size + units, units * 2)).astype(
            np.float32
        )
        g_bias = np.random.uniform(-1.0, 1.0, (units * 2,)).astype(np.float32)
        cKernel = np.random.uniform(-1.0, 1.0, (input_size + units, units)).astype(
            np.float32
        )
        cBias = np.random.uniform(-1.0, 1.0, (units,)).astype(np.float32)
        state = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)

        gi = np.concatenate((inputs, state), axis=1)
        value = self.sigmoid(np.dot(gi, g_kernel) + g_bias)
        r, u = np.split(value, 2, axis=1)

        rState = r * state

        ci = np.concatenate((inputs, rState), axis=1)
        c = np.tanh(np.dot(ci, cKernel) + cBias)
        expect = u * state + (1 - u) * c

        return {
            "units": np.asarray([units], dtype=np.int32),
            "inputs": inputs,
            "state": state,
            "gru_cell/gates/kernel": g_kernel,
            "gru_cell/gates/bias": g_bias,
            "gru_cell/candidate/kernel": cKernel,
            "gru_cell/candidate/bias": cBias,
            "expect": expect,
        }
