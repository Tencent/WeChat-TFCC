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


class PyTorchGRUCell(test.Test):
    def getRandomTestData(self):
        batch = np.random.randint(1, 5)
        units = np.random.randint(64, 512)
        inputSize = np.random.randint(64, 512)

        inputs = np.random.uniform(-1.0, 1.0, (batch, inputSize)).astype(np.float32)
        gates_kernel = np.random.uniform(
            -1.0, 1.0, (inputSize + units, units * 2)
        ).astype(np.float32)
        gates_bias = np.random.uniform(-1.0, 1.0, (units * 2,)).astype(np.float32)
        candidate_i_kernel = np.random.uniform(-1.0, 1.0, (inputSize, units)).astype(
            np.float32
        )
        candidate_i_bias = np.random.uniform(-1.0, 1.0, (units,)).astype(np.float32)
        candidate_h_kernel = np.random.uniform(-1.0, 1.0, (units, units)).astype(
            np.float32
        )
        candidate_h_bias = np.random.uniform(-1.0, 1.0, (units,)).astype(np.float32)
        state = np.random.uniform(-1.0, 1.0, (batch, units)).astype(np.float32)

        gi = np.concatenate([inputs, state], axis=1)
        value = np.dot(gi, gates_kernel) + gates_bias
        value = 1 / (1 + np.exp(-value))
        r = value[:, :units]
        z = value[:, units:]

        ni = np.dot(inputs, candidate_i_kernel) + candidate_i_bias
        nh = np.dot(state, candidate_h_kernel) + candidate_h_bias

        n = np.tanh(ni + r * nh)
        expect = n + z * (state - n)

        return {
            "units": np.asarray([units], dtype=np.int32),
            "gru_cell/weight_gates": gates_kernel,
            "gru_cell/bias_gates": gates_bias,
            "gru_cell/weight_candidate_i": candidate_i_kernel,
            "gru_cell/bias_candidate_i": candidate_i_bias,
            "gru_cell/weight_candidate_h": candidate_h_kernel,
            "gru_cell/bias_candidate_h": candidate_h_bias,
            "inputs": inputs,
            "state": state,
            "expect": expect,
        }
