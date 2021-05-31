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

import enum
import numpy as np
from ir.node import Node
import ir.framework
from ir.common import get_broadcast_shape


class MaxPool(Node):
    class DataFormat(enum.Enum):
        NCHW = 0
        NHWC = 1
        NCW = 2
        NWC = 3

    def update_attributes(self, kernels, pads, strides, data_format):
        assert all([isinstance(v, int) for v in kernels])
        assert all([isinstance(v, int) for v in pads])
        assert all([isinstance(v, int) for v in strides])
        assert len(kernels) == len(pads) and len(pads) == len(strides)
        self._kernels = kernels
        self._pads = pads
        self._strides = strides
        self._data_format = self.DataFormat(data_format)

    def inference(self):
        assert len(self.inputs) == 1
        assert len(self.outputs) == 1
        assert self.inputs[0].is_tensor()
        assert len(self.inputs[0].shape) == 4 or len(self.inputs[0].shape) == 3
        if len(self.inputs[0].shape) == 3:
            assert self.data_format in (self.DataFormat.NCW, self.DataFormat.NWC)
        elif len(self.inputs[0].shape) == 4:
            assert self.data_format in (self.DataFormat.NCHW, self.DataFormat.NHWC)
        else:
            raise RuntimeError("Unknow error")

        if self.data_format in (self.DataFormat.NCW, self.DataFormat.NCHW):
            input_size = self.inputs[0].shape[2:]
        elif self.data_format in (self.DataFormat.NWC, self.DataFormat.NHWC):
            input_size = self.inputs[0].shape[1:-1]
        else:
            raise RuntimeError("Unknow error")

        output_size = []
        for i in range(len(input_size)):
            if isinstance(input_size, int):
                size = (
                    input_size + 2 * self.pads[i] - self.kernels[i]
                ) // self.strides[i] + 1
            else:
                size = self.create_shape_name("max_pool")
            output_size.append(size)

        if self.data_format in (self.DataFormat.NCW, self.DataFormat.NCHW):
            shape = self.inputs[0].shape[:2] + output_size
        elif self.data_format in (self.DataFormat.NWC, self.DataFormat.NHWC):
            shape = [self.inputs[0].shape[0]] + output_size + [self.inputs[0].shape[-1]]
        else:
            raise RuntimeError("Unknow error")

        self.outputs[0].stype = ir.framework.SymbolType.VARIABLE
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[0].shape = shape

    @property
    def kernels(self):
        return self._kernels

    @property
    def pads(self):
        return self._pads

    @property
    def strides(self):
        return self._strides

    @property
    def data_format(self):
        return self._data_format

    @property
    def attributes(self):
        return {
            "kernels": self.kernels,
            "pads": self.pads,
            "strides": self.strides,
            "data_format": self.data_format,
        }
