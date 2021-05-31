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


class ReduceSum(test.Test):
    @property
    def classPrefix(self):
        return "reduce_sum"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)

        inp = np.random.uniform(-1.0, 1.0, shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.sum(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1
        return {
            "input": inp,
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect,
        }


class ReduceMean(test.Test):
    @property
    def classPrefix(self):
        return "reduce_mean"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)
        inp = np.random.uniform(-1.0, 1.0, shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.mean(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1

        return {
            "input": inp,
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect,
        }


class ReduceProd(test.Test):
    @property
    def classPrefix(self):
        return "reduce_prod"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)
        inp = np.random.uniform(-1.0, 1.0, shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.prod(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1

        return {
            "input": inp,
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect,
        }


class ReduceMax(test.Test):
    @property
    def classPrefix(self):
        return "reduce_max"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)
        inp = np.random.uniform(-1.0, 1.0, shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.max(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1

        return {
            "input": inp,
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect,
        }


class ReduceMin(test.Test):
    @property
    def classPrefix(self):
        return "reduce_min"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)
        inp = np.random.uniform(-1.0, 1.0, shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.min(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1

        return {
            "input": inp,
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect,
        }


class ReduceAny(test.Test):
    @property
    def classPrefix(self):
        return "reduce_any"

    def getRandomTestData(self):
        dims = np.random.randint(3, 10)
        shape = np.random.randint(1, 10, dims)
        inp = np.random.choice([True, False], shape)
        keep = np.random.randint(dims)

        expect = inp
        axis = dims - 1
        while axis >= keep:
            expect = expect.any(axis)
            expect = np.expand_dims(expect, axis)
            axis -= 1

        return {
            "input": inp.astype(np.uint8),
            "keep": np.asarray([keep], dtype=np.int32),
            "expect": expect.astype(np.uint8),
        }
