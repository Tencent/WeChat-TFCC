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

# import torch


def get_dilated_kernel(kernel, dilations):
    out_channels, in_channels, k_h, k_w = kernel.shape
    d_h, d_w = dilations
    new_k_h = (k_h - 1) * d_h + 1
    new_k_w = (k_w - 1) * d_w + 1
    result = np.zeros((out_channels, in_channels, new_k_h, new_k_w))
    for out_c in range(0, out_channels):
        for in_c in range(0, in_channels):
            origin_row = 0
            origin_col = 0
            for row in range(0, new_k_h, d_h):
                origin_col = 0
                for col in range(0, new_k_w, d_w):
                    result[out_c, in_c, row, col] = kernel[
                        out_c, in_c, origin_row, origin_col
                    ]
                    origin_col = origin_col + 1
                origin_row = origin_row + 1
    return result


def numpy_conv2d(inputs, kernel, strides, paddings, dilations=[1, 1]):
    out_channels, in_channels, origin_kernel_h, origin_kernel_w = kernel.shape
    batch, in_channels, input_h, input_w = inputs.shape
    padding_h, padding_w = paddings
    dilation_h, dilation_w = dilations
    stride_h, stride_w = strides

    new_kernel = get_dilated_kernel(kernel, dilations)
    new_inputs = np.pad(
        inputs, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w))
    )
    out_channels, in_channels, new_kernel_h, new_kernel_w = new_kernel.shape

    output_h = (
        input_h + 2 * padding_h - dilation_h * (origin_kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + 2 * padding_w - dilation_w * (origin_kernel_w - 1) - 1
    ) // stride_w + 1

    result = np.zeros((batch, out_channels, output_h, output_w))
    for b in range(0, batch):
        for out_c in range(0, out_channels):
            input_r = 0
            input_c = 0
            for row in range(0, output_h):
                for col in range(0, output_w):
                    for in_c in range(0, in_channels):
                        cur_input = new_inputs[
                            b,
                            in_c,
                            input_r : input_r + new_kernel_h,
                            input_c : input_c + new_kernel_w,
                        ]
                        cur_output = cur_input * new_kernel[out_c, in_c]
                        conv_sum = np.sum(cur_output)
                        result[b, out_c, row, col] = (
                            result[b, out_c, row, col] + conv_sum
                        )
                    input_c = input_c + stride_w
                input_r = input_r + stride_h
                input_c = 0
    return result


# i_h = 5
# i_w = 5
# k_h = 3
# k_w = 3
# strides = [2, 2]
# dilations = [2, 2]
# paddings = [2, 2]

# inp = np.random.randint(1, 10, [1, 1, 100, 100])
# ker = np.random.randint(1, 10, [1, 1, 3, 3])
# torch_inp = torch.from_numpy(inp)
# torch_ker = torch.from_numpy(ker)
# print((inp == torch_inp.numpy()).all())
# print((ker == torch_ker.numpy()).all())
# print(inp)
# print(ker)
# np_out = numpy_conv2d(inp, ker, [1, 1], [2, 2],[1, 1])
# torch_out = torch.nn.functional.conv2d(torch_inp, torch_ker, stride = 1, padding = 2, dilation = 1)
# m = torch.nn.Conv2d(1, 1, 3, stride=1, padding=2,bias=False)
# torch_out = m(torch.from_numpy(inp))
# print(np_out.shape)
# print(torch_out.numpy().shape)
# print((np_out == torch_out.numpy()).all())
