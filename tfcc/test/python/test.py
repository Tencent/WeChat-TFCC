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


import numpy as np


class Test(object):
    @property
    def modulePrefix(self):
        return self.__class__.__module__.split(".")[-2].lower()

    @property
    def classPrefix(self):
        return self.__class__.__name__.lower()

    @property
    def prefix(self):
        return self.modulePrefix + "/" + self.classPrefix

    @property
    def randomLoopCount(self):
        return 5

    def getSpecialTestData(self):
        return []

    def getRandomTestData(self):
        return {}

    def getData(self):
        datas = self.getSpecialTestData() + [
            self.getRandomTestData() for i in range(self.randomLoopCount)
        ]
        currentIndex = 2
        while True:
            funcName = "getRandomTestData" + str(currentIndex)
            if not hasattr(self, funcName) or not callable(getattr(self, funcName)):
                break
            loop = self.randomLoopCount
            if hasattr(self, "randomLoopCount" + str(currentIndex)):
                loop = int(getattr(self, "randomLoopCount" + str(currentIndex)))
            datas = datas + [getattr(self, funcName)() for i in range(loop)]
            currentIndex += 1
        result = {self.prefix + "/__count__": np.asarray(len(datas), dtype=np.int32)}
        for i, v in enumerate(datas):
            for k in v:
                result[self.prefix + "/" + str(i) + "/" + k] = v[k]
        return result
