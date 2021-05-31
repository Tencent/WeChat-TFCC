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


class Context(object):
    def __init__(self):
        self._idx = 0
        self._frontend_name = ""
        self._backend_name = ""
        self._frontend = {}
        self._backend = {}
        self._graph_names = set()

    def create_shape_name(self, prefix: str):
        current_idx = self._idx
        self._idx += 1

        return prefix + "_" + str(current_idx)

    def create_symbol_name(self, prefix: str):
        current_idx = self._idx
        self._idx += 1
        return prefix + "_" + str(current_idx)

    def create_graph_name(self, prefix: str):
        name = prefix
        while name in self._graph_names:
            current_idx = self._idx
            self._idx += 1
            name = prefix + "_" + str(current_idx)
        self._graph_names.add(name)
        return name

    @property
    def frontend_name(self):
        return self._frontend_name

    @frontend_name.setter
    def frontend_name(self, value: str):
        self._frontend_name = value

    @property
    def backend_name(self):
        return self._backend_name

    @backend_name.setter
    def backend_name(self, value: str):
        self._backend_name = value

    @property
    def frontend(self):
        return self._frontend.copy()

    @frontend.setter
    def frontend(self, value: dict):
        self._frontend = value

    @property
    def backend(self):
        return self._backend.copy()

    @backend.setter
    def backend(self, value: dict):
        self._backend = value
