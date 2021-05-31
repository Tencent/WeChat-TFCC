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


def get_all_converters(op_set: dict):
    from .cast import Cast
    from .concat import Concat
    from .constant import Constant
    from .constantofshape import ConstantOfShape
    from .expand import Expand
    from .eyelike import EyeLike
    from .flatten import Flatten
    from .identity import Identity
    from .if_ import If
    from .loop import Loop
    from .pad import Pad
    from .range import Range
    from .reshape import Reshape
    from .shape import Shape
    from .slice import Slice
    from .split import Split
    from .squeeze import Squeeze
    from .tile import Tile
    from .transpose import Transpose
    from .unsqueeze import Unsqueeze

    return [
        Cast(op_set),
        Concat(op_set),
        Constant(op_set),
        ConstantOfShape(op_set),
        Expand(op_set),
        EyeLike(op_set),
        Flatten(op_set),
        Identity(op_set),
        If(op_set),
        Loop(op_set),
        Pad(op_set),
        Range(op_set),
        Reshape(op_set),
        Shape(op_set),
        Slice(op_set),
        Split(op_set),
        Squeeze(op_set),
        Tile(op_set),
        Transpose(op_set),
        Unsqueeze(op_set),
    ]
