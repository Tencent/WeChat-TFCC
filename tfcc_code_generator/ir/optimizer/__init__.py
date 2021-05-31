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


def get_all_optimizers():
    from .patternmanager import get_all_pattern_managers

    optimizers = get_all_pattern_managers()

    from . import base

    modules = [
        base,
    ]

    for module in modules:
        optimizers += module.get_all_optimizers()
    return optimizers
