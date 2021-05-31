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


def get_all_node_generators():
    from .averagepool import AveragePool
    from .batchnormalization import BatchNormalization
    from .conv import Conv
    from .gather import Gather
    from .globalaveragepool import GlobalAveragePool
    from .layernormalization import LayerNormalization
    from .localresponsenormalization import LocalResponseNormalization
    from .maxpool import MaxPool
    from .nonzero import NonZero
    from .onehot import OneHot
    from .scatterndwithdata import ScatterNDWithData
    from .where import Where

    return [
        AveragePool,
        BatchNormalization,
        Conv,
        Gather,
        GlobalAveragePool,
        LayerNormalization,
        LocalResponseNormalization,
        MaxPool,
        NonZero,
        OneHot,
        ScatterNDWithData,
        Where,
    ]
