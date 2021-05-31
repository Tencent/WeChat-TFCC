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

from tensorflow.python.framework.ops import Operation
import ir.framework
import ir.node


class Converter(object):
    def __init__(self):
        pass

    def accept(self, op: Operation):
        if op.type != self.op_type:
            return False
        return True

    def may_accept(self, op: Operation):
        return op.type == self.op_type

    @property
    def op_type(self) -> str:
        return self.__class__.__name__.split(".")[-1]
