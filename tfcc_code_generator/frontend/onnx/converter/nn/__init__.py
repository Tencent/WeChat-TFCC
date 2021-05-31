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
    from .averagepool import AveragePool
    from .batchnormalization import BatchNormalization
    from .conv import Conv
    from .dropout import Dropout
    from .gather import Gather
    from .globalaveragepool import GlobalAveragePool
    from .localresponsenormalization import LocalResponseNormalization
    from .maxpool import MaxPool
    from .nonzero import NonZero
    from .onehot import OneHot
    from .scatternd import ScatterND
    from .where import Where

    return [
        AveragePool(op_set),
        BatchNormalization(op_set),
        Conv(op_set),
        Dropout(op_set),
        Gather(op_set),
        GlobalAveragePool(op_set),
        LocalResponseNormalization(op_set),
        MaxPool(op_set),
        NonZero(op_set),
        OneHot(op_set),
        ScatterND(op_set),
        Where(op_set),
    ]
