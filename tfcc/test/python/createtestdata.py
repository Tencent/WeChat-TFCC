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


import os
import argparse
import numpy as np

import testcase


def getTestNameList():
    result = []
    allModules = testcase.allModules
    for mlo in allModules:
        for cls in mlo.allTests:
            t = cls()
            result.append(t.prefix)
    result.sort()
    return result


def getModuleData(mlo):
    result = {}
    moduleName = mlo.__name__.split(".")[-1]
    print("==== " + moduleName + " ====")
    for cls in mlo.allTests:
        t = cls()
        tr = t.getData()
        result.update(tr)
        print(t.__class__.__name__)
    return result


def getData():
    result = {}
    allModules = testcase.allModules
    for mlo in allModules:
        moduleName = mlo.__name__.split(".")[-1]
        tr = getModuleData(mlo)
        result.update(tr)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=str, required=True, help="Full path to output file path"
    )
    parser.add_argument(
        "--force_update", type=str, default="True", help="replace the old file if exist"
    )

    args = parser.parse_args()
    nameList = getTestNameList()

    if args.force_update == "False":
        if os.path.exists(args.out + ".npz"):
            oldData = np.load(args.out + ".npz")
            if (
                "__test_name_list__" in oldData
                and oldData["__test_name_list__"].tolist() == nameList
            ):
                print("data has existed. skip.")
                return
            else:
                print("data need update.")

    result = getData()
    result["__test_name_list__"] = np.asarray(nameList)
    np.savez(args.out, **result)


if __name__ == "__main__":
    main()
