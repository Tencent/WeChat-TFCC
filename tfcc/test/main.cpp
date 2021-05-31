// Copyright 2021 Wechat Group, Tencent
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "environment.h"

int showUsage(const std::string& path) {
  std::cout << "Usage: " << path << " [cuda/mkl] [test data path] [gtest options...]" << std::endl;
  char* nArgv[2];
  char hv[] = "-h";
  nArgv[0] = const_cast<char*>(path.c_str());
  nArgv[1] = hv;
  int nArgc = 2;
  testing::InitGoogleTest(&nArgc, nArgv);
  return RUN_ALL_TESTS();
}

GTEST_API_ int main(int argc, char** argv) {
  std::string device = "CUDA";
  if (argc < 3) {
    return showUsage(argv[0]);
  }
  if (std::string(argv[1]) == std::string("mkl")) {
    Environment::setDefaultDeviceType(tfcc::DeviceType::MKL);
    device = "MKL";
  } else if (std::string(argv[1]) == std::string("cuda")) {
    Environment::setDefaultDeviceType(tfcc::DeviceType::CUDA);
    device = "CUDA";
  } else if (std::string(argv[1]) == std::string("all")) {
    device = "ALL";
  } else {
    return showUsage(argv[0]);
  }

  Environment::setDefaultTestDataPath(argv[2]);

  std::cout << "Running main() from " << __FILE__ << " with device " << device << std::endl;
  testing::InitGoogleTest(&argc, argv);
  if (device == "ALL") {
    int ret = 0;
#ifdef TFCC_WITH_MKL
    Environment::setDefaultDeviceType(tfcc::DeviceType::MKL);
    ret += RUN_ALL_TESTS();
#endif
#ifdef TFCC_WITH_CUDA
    Environment::setDefaultDeviceType(tfcc::DeviceType::CUDA);
    ret += RUN_ALL_TESTS();
#endif
    return ret;
  }
  return RUN_ALL_TESTS();
}
