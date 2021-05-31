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


def get_all_converters():
    from .assert_ import Assert
    from .cast import Cast
    from .concat import Concat
    from .concatv2 import ConcatV2
    from .expanddims import ExpandDims
    from .fill import Fill
    from .matrixbandpart import MatrixBandPart
    from .identity import Identity
    from .pad import Pad
    from .pack import Pack
    from .shape import Shape
    from .slice import Slice
    from .split import Split
    from .splitv import SplitV
    from .squeeze import Squeeze
    from .stopgradient import StopGradient
    from .stridedslice import StridedSlice
    from .range import Range
    from .reshape import Reshape
    from .transpose import Transpose
    from .tile import Tile
    from .zeroslike import ZerosLike

    return [
        Assert(),
        Cast(),
        Concat(),
        ConcatV2(),
        ExpandDims(),
        Fill(),
        Identity(),
        MatrixBandPart(),
        Pad(),
        Pack(),
        Shape(),
        Slice(),
        Split(),
        SplitV(),
        Squeeze(),
        StopGradient(),
        StridedSlice(),
        Range(),
        Reshape(),
        Transpose(),
        Tile(),
        ZerosLike(),
    ]
