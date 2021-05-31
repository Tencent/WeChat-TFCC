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



from .slice import Slice
from .concat import Concat
from .stack import Stack
from .transpose import Transpose
from .tril import Tril
from .triu import Triu
from .unstack import Unstack

allTests = [Slice, Concat, Stack, Transpose, Tril, Triu, Unstack]
