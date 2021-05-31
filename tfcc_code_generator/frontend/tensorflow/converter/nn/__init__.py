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
    from .conv2d import Conv2D
    from .depthwiseconv2dnative import DepthwiseConv2dNative
    from .fusedbatchnormv3 import FusedBatchNormV3
    from .gather import Gather
    from .gatherv2 import GatherV2
    from .maxpool import MaxPool
    from .onehot import OneHot
    from .select import Select

    return [
        Conv2D(),
        DepthwiseConv2dNative(),
        FusedBatchNormV3(),
        Gather(),
        GatherV2(),
        MaxPool(),
        OneHot(),
        Select(),
    ]
