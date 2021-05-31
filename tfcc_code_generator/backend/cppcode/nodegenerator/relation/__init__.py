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
    from .equal import Equal
    from .less import Less
    from .lessorequal import LessOrEqual
    from .greater import Greater
    from .greaterorequal import GreaterOrEqual
    from .not_ import Not
    from .unequal import UnEqual

    return [
        Equal,
        Less,
        LessOrEqual,
        Greater,
        GreaterOrEqual,
        Not,
        UnEqual,
    ]
