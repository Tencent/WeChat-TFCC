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


def get_test_data(has_data: bool, dims: int):
    assert dims >= 1
    indices_size = np.random.randint(0, dims) + 1
    shape = np.random.randint(1, 10, dims)
    if has_data:
        data = np.random.uniform(-1.0, 1.0, shape)
    else:
        data = np.zeros(shape)

    indices_dims = np.random.randint(0, indices_size) + 2
    indices_shape = np.random.permutation(shape[:indices_size])[: indices_dims - 1]
    indices_shape = [np.random.randint(0, s) + 1 for s in indices_shape] + [
        indices_size
    ]

    indices = list(np.ndindex(tuple(shape[:indices_size])))
    indices = np.random.permutation(indices)[: np.prod(indices_shape[:-1])]
    indices = np.asarray(indices).reshape(indices_shape)

    updates_shape = list(indices_shape[:-1]) + list(shape[indices_size:])
    updates = np.random.uniform(-1.0, 1.0, updates_shape)

    expect = np.copy(data)
    update_indices = indices.shape[:-1]
    for idx in np.ndindex(update_indices):
        expect[tuple(indices[idx])] = updates[idx]

    return {
        "has_data": 1 if has_data else 0,
        "data": data,
        "indices": indices,
        "updates": updates,
        "expect": expect,
    }


class ScatterND(test.Test):
    @property
    def classPrefix(self):
        return "scatter_nd"

    def getRandomTestData(self):
        return get_test_data(True, 1)

    def getRandomTestData2(self):
        return get_test_data(True, 2)

    def getRandomTestData3(self):
        return get_test_data(True, 3)

    def getRandomTestData4(self):
        return get_test_data(True, 4)

    def getRandomTestData5(self):
        return get_test_data(True, 5)

    def getRandomTestData6(self):
        return get_test_data(False, 1)

    def getRandomTestData7(self):
        return get_test_data(False, 2)

    def getRandomTestData8(self):
        return get_test_data(False, 3)

    def getRandomTestData9(self):
        return get_test_data(False, 4)

    def getRandomTestData10(self):
        return get_test_data(False, 5)
