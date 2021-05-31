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
from ir.node import Node
import ir.framework
from ir.common import get_broadcast_shape


class Conv(Node):
    def update_attributes(
        self,
        padding_height: int,
        padding_width: int,
        stride_height: int,
        stride_width: int,
        dilate_height: int,
        dilate_width: int,
        group: int,
        nchw: bool,
    ):
        assert isinstance(padding_height, int) and padding_height >= 0
        self._padding_height = padding_height
        assert isinstance(padding_width, int) and padding_width >= 0
        self._padding_width = padding_width

        assert isinstance(stride_height, int) and stride_height >= 0
        self._stride_height = stride_height
        assert isinstance(stride_width, int) and stride_width >= 0
        self._stride_width = stride_width

        assert isinstance(dilate_height, int) and dilate_height >= 1
        self._dilate_height = dilate_height
        assert isinstance(dilate_width, int) and dilate_width >= 1
        self._dilate_width = dilate_width

        assert isinstance(group, int) and group >= 1
        self._group = group

        assert isinstance(nchw, bool)
        self._nchw = nchw

    def inference(self):
        assert len(self.inputs) == 2
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert len(self.inputs[0].shape) == 4
        assert self.inputs[1].is_tensor()
        assert len(self.inputs[1].shape) == 4
        assert self.inputs[0].dtype == self.inputs[1].dtype

        if self.nchw:
            n, c, h, w = self.inputs[0].shape
        else:
            n, h, w, c = self.inputs[0].shape

        if isinstance(c, int) and isinstance(self.inputs[1].shape[1], int):
            assert c == self.inputs[1].shape[1] * self.group

        if isinstance(h, int) and isinstance(self.inputs[1].shape[2], int):
            kernel_height = self.inputs[1].shape[2]
            new_h = (
                h
                + 2 * self.padding_height
                - self.dilate_height * (kernel_height - 1)
                - 1
            ) // self.stride_height + 1
        else:
            new_h = self.create_shape_name("conv")
        if isinstance(w, int) and isinstance(self.inputs[1].shape[3], int):
            kernel_width = self.inputs[1].shape[3]
            new_w = (
                w + 2 * self.padding_width - self.dilate_width * (kernel_width - 1) - 1
            ) // self.stride_width + 1
        else:
            new_w = self.create_shape_name("conv")

        output_channels = self.inputs[1].shape[0]

        if self.nchw:
            shape = [n, output_channels, new_h, new_w]
        else:
            shape = [n, new_h, new_w, output_channels]

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def padding_height(self):
        return self._padding_height

    @property
    def padding_width(self):
        return self._padding_width

    @property
    def stride_height(self):
        return self._stride_height

    @property
    def stride_width(self):
        return self._stride_width

    @property
    def dilate_height(self):
        return self._dilate_height

    @property
    def dilate_width(self):
        return self._dilate_width

    @property
    def group(self):
        return self._group

    @property
    def nchw(self):
        return self._nchw

    @property
    def attributes(self):
        return {
            "padding_height": self.padding_height,
            "padding_width": self.padding_width,
            "stride_height": self.stride_height,
            "stride_width": self.stride_width,
            "dilate_height": self.dilate_height,
            "dilate_width": self.dilate_width,
            "group": self.group,
            "nchw": self.nchw,
        }
