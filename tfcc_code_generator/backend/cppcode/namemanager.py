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

import re


class NameManager(object):
    def __init__(self):
        self._objectMap = {}
        self._nameSet = set(["s_", "s_tmp", "i", "j", "k", "s_loop_result"])
        self._idx = 1

    def set_symbol_name(self, obj, name: str):
        assert isinstance(name, str) and len(name) > 0
        assert re.match(r"^[\w_][\w\d_]*$", name)
        assert name not in self._nameSet
        assert obj not in self._objectMap

        self._objectMap[obj] = name
        self._nameSet.add(name)

    def get_symbol_name(self, obj, name: str = None):
        if obj is not None and obj in self._objectMap:
            return self._objectMap[obj]

        if not name:
            name = "s_"

        name = re.sub(r"[^\w\d_]", "_", name)
        if re.match(r"^\d", name):
            name = "s_" + name
        real_name = name
        while real_name in self._nameSet:
            real_name = name + str(self._idx)
            self._idx += 1

        if obj is not None:
            self._objectMap[obj] = real_name
        self._nameSet.add(real_name)
        return real_name
