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


# ICFO kernel format
class BidirectionalLSTM(Node):
    def update_attributes(self, hidden_size):
        assert isinstance(hidden_size, int) and hidden_size > 0
        self._hidden_size = hidden_size

    def inference(self):
        assert len(self.inputs) == 6
        assert len(self.outputs) == 6
        dtype = self.inputs[0].dtype
        assert all([dtype == inp.dtype for inp in self.inputs])
        assert all([inp.is_tensor() for inp in self.inputs])

        # input_symbol: [seq_length, batch_size, input_size]
        # input_kernel_symbol: [2, input_size, 4 * hidden_size]
        # state_kernel_symbol: [2, hidden_size, 4 * hidden_size]
        # bias_symbol: [2, 4 * hidden_size]
        # initial_h: [2, batch_size, hidden_size]
        # initial_c: [2, batch_size, hidden_size]
        (
            input_symbol,
            input_kernel_symbol,
            state_kernel_symbol,
            bias_symbol,
            initial_h,
            initial_c,
        ) = self.inputs

        assert len(input_symbol.shape) == 3
        seq_length, batch_size, input_size = input_symbol.shape

        # check input kernel
        assert len(input_kernel_symbol.shape) == 3
        assert (
            isinstance(input_kernel_symbol.shape[0], str)
            or input_kernel_symbol.shape[0] == 2
        )
        if isinstance(input_size, str):
            if isinstance(input_kernel_symbol.shape[1], int):
                input_size = input_kernel_symbol.shape[1]
        else:
            assert (
                isinstance(input_kernel_symbol.shape[1], str)
                or input_kernel_symbol.shape[1] == input_size
            )
        assert (
            isinstance(input_kernel_symbol.shape[2], str)
            or input_kernel_symbol.shape[2] == self.hidden_size * 4
        )

        # check state kernel
        assert len(state_kernel_symbol.shape) == 3
        assert (
            isinstance(state_kernel_symbol.shape[0], str)
            or state_kernel_symbol.shape[0] == 2
        )
        assert (
            isinstance(state_kernel_symbol.shape[1], str)
            or state_kernel_symbol.shape[1] == self.hidden_size
        )
        assert (
            isinstance(state_kernel_symbol.shape[2], str)
            or state_kernel_symbol.shape[2] == self.hidden_size * 4
        )

        # check bias
        assert len(bias_symbol.shape) == 2
        assert isinstance(bias_symbol.shape[0], str) or bias_symbol.shape[0] == 2
        assert (
            isinstance(bias_symbol.shape[1], str)
            or bias_symbol.shape[1] == self.hidden_size * 4
        )

        # check initial h
        assert len(initial_h.shape) == 3
        assert initial_h.shape[0] == 2
        if isinstance(batch_size, str):
            if isinstance(initial_h.shape[1], int):
                batch_size = initial_h.shape[1]
        else:
            assert (
                isinstance(initial_h.shape[1], str) or initial_h.shape[1] == batch_size
            )
        assert (
            isinstance(initial_h.shape[2], str)
            or initial_h.shape[2] == self.hidden_size
        )

        # check initial c
        assert len(initial_c.shape) == 3
        assert initial_c.shape[0] == 2
        if isinstance(batch_size, str):
            if isinstance(initial_c.shape[1], int):
                batch_size = initial_c.shape[1]
        else:
            assert (
                isinstance(initial_c.shape[1], str) or initial_c.shape[1] == batch_size
            )
        assert (
            isinstance(initial_c.shape[2], str)
            or initial_c.shape[2] == self.hidden_size
        )

        # forward_output_symbol: [seq_length, batch_size, hidden_size]
        # backword_output_symbol: [seq_length, batch_size, hidden_size]
        # forward_h: [batch_size, hidden_size]
        # backword_h: [batch_size, hidden_size]
        # forward_c: [batch_size, hidden_size]
        # backword_c: [batch_size, hidden_size]
        (
            forward_output_symbol,
            backword_output_symbol,
            forward_h,
            backword_h,
            forward_c,
            backword_c,
        ) = self.outputs
        forward_output_symbol.stype = ir.framework.SymbolType.VARIABLE
        forward_output_symbol.dtype = dtype
        forward_output_symbol.shape = [seq_length, batch_size, self.hidden_size]

        backword_output_symbol.stype = ir.framework.SymbolType.VARIABLE
        backword_output_symbol.dtype = dtype
        backword_output_symbol.shape = [seq_length, batch_size, self.hidden_size]

        forward_h.stype = ir.framework.SymbolType.VARIABLE
        forward_h.dtype = dtype
        forward_h.shape = [batch_size, self.hidden_size]

        backword_h.stype = ir.framework.SymbolType.VARIABLE
        backword_h.dtype = dtype
        backword_h.shape = [batch_size, self.hidden_size]

        forward_c.stype = ir.framework.SymbolType.VARIABLE
        forward_c.dtype = dtype
        forward_c.shape = [batch_size, self.hidden_size]

        backword_c.stype = ir.framework.SymbolType.VARIABLE
        backword_c.dtype = dtype
        backword_c.shape = [batch_size, self.hidden_size]

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def attributes(self):
        return {"hidden_size": self.hidden_size}
