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
from .convolution_impl import numpy_conv2d


class Conv(test.Test):
    def getRandomTestData(self):
        input_size = np.random.randint(40, 70)
        kernel_size = np.random.randint(1, 15)
        bath = np.random.randint(1, 10)
        in_channel = np.random.randint(1, 5)
        out_channel = np.random.randint(3, 10)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

        inputs = np.random.uniform(0, 100, [bath, in_channel, input_size, input_size])
        kernel = np.random.uniform(
            0, 100, [out_channel, in_channel, kernel_size, kernel_size]
        )

        paddings = np.random.randint(0, 50, 2)
        strides = np.random.randint(1, 4, 2)
        dilations = np.random.randint(1, 4, 2)

        expect = numpy_conv2d(inputs, kernel, strides, paddings, dilations)

        return {
            "inputs": inputs,
            "kernel": kernel,
            "padding_height": paddings[0],
            "padding_width": paddings[1],
            "stride_height": strides[0],
            "stride_width": strides[1],
            "dilation_height": dilations[0],
            "dilation_width": dilations[1],
            "expect": expect,
        }
