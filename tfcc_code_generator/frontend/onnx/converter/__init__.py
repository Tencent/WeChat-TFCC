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
    from . import base
    from . import extensions
    from . import math
    from . import nn
    from . import random
    from . import relation
    from . import rnn

    modules = [base, extensions, math, nn, random, relation, rnn]
    converters = []
    for module in modules:
        converters += module.get_all_converters(op_set)
    converters = list(filter(lambda x: x.valid, converters))
    return converters
