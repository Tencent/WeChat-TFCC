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
#include <limits>
#include <string>
#include <vector>

#include "tfcc.h"

#include "environment.h"

TEST(UtilsTest, unzip) {
  {
    std::vector<std::string> nameList = {
        "d:0.npy",
        "model/a:0.npy",
        "model/test/b:0.npy",
        "model/test/c:0.npy",
        "model/test/e:0.npy",
    };
    auto dataMap =
        tfcc::unzip_from_path(Environment::getDefaultTestDataPath() + "/npz_testfile.npz");
    ASSERT_EQ(dataMap.size(), nameList.size());
    for (std::string name : nameList) {
      auto it = dataMap.find(name);
      ASSERT_NE(it, dataMap.end()) << name << " not in name map" << std::endl;
    }
  }

  {
    std::vector<std::string> nameList = {
        "d:0.npy",
        "model/a:0.npy",
        "model/test/b:0.npy",
        "model/test/c:0.npy",
    };

    auto dataMap = tfcc::unzip_from_path(
        Environment::getDefaultTestDataPath() + "/npz_testfile_compressed.npz");
    ASSERT_EQ(dataMap.size(), nameList.size());
    for (std::string name : nameList) {
      auto it = dataMap.find(name);
      ASSERT_NE(it, dataMap.end()) << name << " not in name map" << std::endl;
    }
  }
}

TEST(UtilsTest, parse_npy) {
  std::vector<std::string> nameList = {
      "d:0.npy", "model/a:0.npy", "model/test/b:0.npy", "model/test/c:0.npy", "model/test/e:0.npy"};

  auto dataMap = tfcc::unzip_from_path(Environment::getDefaultTestDataPath() + "/npz_testfile.npz");
  ASSERT_EQ(dataMap.size(), nameList.size());

  // d:0.npy
  {
    auto it = dataMap.find("d:0.npy");
    ASSERT_NE(it, dataMap.end());
    tfcc::Shape s;
    std::string type, data;
    std::tie(s, type, data) = tfcc::parse_npy(it->second);

    ASSERT_EQ(type, "i2");
    ASSERT_EQ(s.size(), 2u);
    ASSERT_EQ(s[0], 2u);
    ASSERT_EQ(s[1], 1u);
    ASSERT_EQ(data.size(), s.area() * sizeof(int16_t));
    const int16_t* p = reinterpret_cast<const int16_t*>(data.data());
    for (size_t i = 0; i < s.area(); ++i) {
      ASSERT_EQ(p[i], static_cast<int>(i + 1));
    }
  }

  // model/a:0.npy
  {
    auto it = dataMap.find("model/a:0.npy");
    ASSERT_NE(it, dataMap.end());
    tfcc::Shape s;
    std::string type, data;
    std::tie(s, type, data) = tfcc::parse_npy(it->second);

    ASSERT_EQ(type, "f8");
    ASSERT_EQ(s.size(), 2u);
    ASSERT_EQ(s[0], 2u);
    ASSERT_EQ(s[1], 3u);
    ASSERT_EQ(data.size(), s.area() * sizeof(double));
    const double* p = reinterpret_cast<const double*>(data.data());
    for (size_t i = 0; i < s.area(); ++i) {
      ASSERT_EQ(p[i], static_cast<double>(i + 1));
    }
  }

  // model/test/b:0.npy
  {
    auto it = dataMap.find("model/test/b:0.npy");
    ASSERT_NE(it, dataMap.end());
    tfcc::Shape s;
    std::string type, data;
    std::tie(s, type, data) = tfcc::parse_npy(it->second);

    ASSERT_EQ(type, "f4");
    ASSERT_EQ(s.size(), 2u);
    ASSERT_EQ(s[0], 2u);
    ASSERT_EQ(s[1], 4u);
    ASSERT_EQ(data.size(), s.area() * sizeof(float));
    const float* p = reinterpret_cast<const float*>(data.data());
    for (size_t i = 0; i < s.area(); ++i) {
      ASSERT_EQ(p[i], static_cast<float>(i + 1));
    }
  }

  // model/test/c:0.npy
  {
    auto it = dataMap.find("model/test/c:0.npy");
    ASSERT_NE(it, dataMap.end());
    tfcc::Shape s;
    std::string type, data;
    std::tie(s, type, data) = tfcc::parse_npy(it->second);

    ASSERT_EQ(type, "i4");
    ASSERT_EQ(s.size(), 2u);
    ASSERT_EQ(s[0], 2u);
    ASSERT_EQ(s[1], 5u);
    ASSERT_EQ(data.size(), s.area() * sizeof(int32_t));
    const int32_t* p = reinterpret_cast<const int32_t*>(data.data());
    for (size_t i = 0; i < s.area(); ++i) {
      ASSERT_EQ(p[i], static_cast<int>(i + 1));
    }
  }

  // model/test/c:0.npy
  {
    auto it = dataMap.find("model/test/e:0.npy");
    ASSERT_NE(it, dataMap.end());
    tfcc::Shape s;
    std::string type, data;
    std::tie(s, type, data) = tfcc::parse_npy(it->second);

    ASSERT_EQ(type, "f8");
    ASSERT_EQ(s.size(), 4u);
    ASSERT_EQ(s[0], 2u);
    ASSERT_EQ(s[1], 3u);
    ASSERT_EQ(s[2], 4u);
    ASSERT_EQ(s[3], 5u);
    ASSERT_EQ(data.size(), s.area() * sizeof(double));
    const double* p = reinterpret_cast<const double*>(data.data());
    for (size_t i = 0; i < s.area(); ++i) {
      ASSERT_EQ(p[i], static_cast<double>(i + 1));
    }
  }
}

TEST(UtilsTest, host_transpose) {
  Environment env;
  env.init();
  for (size_t dims = 2; dims < 6; ++dims) {
    tfcc::Variable<float> v1(makeSequence(3u, 1u, dims));
    tfcc::data::set(v1, makeSequence(1.0f, 1.0f, v1.size()));
    std::vector<size_t> perm = makeSequence(size_t(0), size_t(1), dims);
    std::sort(perm.begin(), perm.end());
    do {
      auto result = tfcc::base::transpose(v1, perm);
      auto v1Data = tfcc::data::get(v1);
      auto hostResult = tfcc::host_transpose(v1Data, v1.shape(), perm);
      tfcc::Variable<float> newResult(result.shape());
      tfcc::data::set(newResult, hostResult);
      ASSERT_TRUE(tfcc::is_similar(result, newResult));
    } while (std::next_permutation(perm.begin(), perm.end()));
  }
  env.release();
}
