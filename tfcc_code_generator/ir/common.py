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
import typing
from ir.framework.context import Context


def get_broadcast_shape(shapes: typing.List[typing.List[int]], context: Context):
    assert len(shapes) > 0
    assert all([x is not None for x in shapes])
    result = shapes[0]
    for shape in shapes[1:]:
        if len(shape) > len(result):
            shape, result = result, shape
        shape = [1 for _ in range(len(result) - len(shape))] + shape
        new_result = []
        assert len(result) == len(shape)
        for a, b in zip(result, shape):
            t = None
            if a == b:
                t = a
            elif a == 1:
                t = b
            elif b == 1:
                t = a
            elif isinstance(a, str) and isinstance(b, int) and b > 1:
                t = b
            elif isinstance(b, str) and isinstance(a, int) and a > 1:
                t = a
            elif isinstance(a, str) and isinstance(b, str):
                t = context.create_shape_name("max_" + a + "_" + b)
            else:
                raise RuntimeError("invalid shape broadcast")
            new_result.append(t)
        result = new_result
    return result


def create_constant_symbol(
    graph, name: str, dtype, stype, shape: list, data: np.ndarray
):
    from ir.framework.symbol import Symbol

    symbol = Symbol(name)
    symbol.dtype = dtype
    symbol.stype = stype
    symbol.origin_stype = stype
    symbol.shape = shape
    symbol.data = data
    assert symbol.verify()
    assert symbol.is_constant()
    graph.add_symbol(symbol)
    graph.add_keep(symbol.name)
