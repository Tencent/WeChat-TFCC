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


class BatchNormalization(test.Test):
    @property
    def classPrefix(self):
        return "batch_normalization"

    def getRandomTestData(self):
        dims = np.random.randint(3, 5)
        shape = np.random.randint(1, 10, dims)
        param_shape = np.copy(shape).tolist()
        for i in range(0, dims):
            if i != 1:
                param_shape[i] = 1
        a = np.random.uniform(-10, 10, shape)
        scale = np.random.uniform(-1.0, 1.0, shape[1])
        offset = np.random.uniform(-1.0, 1.0, shape[1])
        mean = np.random.uniform(-1.0, 1.0, shape[1])
        var = np.random.uniform(0.01, 1.0, shape[1])
        np_scale = np.reshape(scale, param_shape)
        np_offset = np.reshape(offset, param_shape)
        np_mean = np.reshape(mean, param_shape)
        np_var = np.reshape(var, param_shape)

        inv = 1.0 / np.sqrt(np_var + 1e-05) * np_scale
        expect = inv * (a - np_mean) + np_offset

        return {
            "a": a,
            "scale": scale,
            "offset": offset,
            "mean": mean,
            "var": var,
            "epsilon": 1e-05,
            "expect": expect,
        }

    def getRandomTestData2(self):
        dims = 2
        shape = np.random.randint(1, 10, dims)
        param_shape = np.copy(shape).tolist()
        for i in range(0, dims):
            if i != 1:
                param_shape[i] = 1
        a = np.random.uniform(-10, 10, shape)
        scale = np.random.uniform(-1.0, 1.0, shape[1])
        offset = np.random.uniform(-1.0, 1.0, shape[1])
        mean = np.random.uniform(-1.0, 1.0, shape[1])
        var = np.random.uniform(0.01, 1.0, shape[1])
        np_scale = np.reshape(scale, param_shape)
        np_offset = np.reshape(offset, param_shape)
        np_mean = np.reshape(mean, param_shape)
        np_var = np.reshape(var, param_shape)

        inv = 1.0 / np.sqrt(np_var + 1e-05) * np_scale
        expect = inv * (a - np_mean) + np_offset

        return {
            "a": a,
            "scale": scale,
            "offset": offset,
            "mean": mean,
            "var": var,
            "epsilon": 1e-05,
            "expect": expect,
        }
