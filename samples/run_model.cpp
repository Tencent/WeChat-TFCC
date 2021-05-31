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

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "tfcc.h"
#include "tfcc_mkl.h"

#include "tfcc_runtime/tfcc_runtime.h"

std::string load_data_from_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("open file: \"" + path + "\" failed");
  }
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return data;
}

int main(int argc, char* argv[]) {
  tfcc::runtime::Graph::setRecordDetailErrorThreadLocal(true);

  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " [model path]" << std::endl;
    return 1;
  }
  tfcc::initialize_mkl(1, 4);
  std::string modelData = load_data_from_file(argv[1]);
  std::string testData = load_data_from_file(argv[2]);

  tfcc::MultiDataLoader loader;
  tfcc::DataLoader::setGlobalDefault(&loader);

  tfcc::Coster coster;
  tfcc::runtime::Model model(modelData);
  std::cout << "Load model cost: " << coster.lap().milliseconds() << std::endl;

  tfcc::runtime::data::Inputs inputs;
  tfcc::runtime::data::Outputs outputs;

  // set inputs
  auto item = inputs.add_items();
  item->set_name("The input name");
  item->set_dtype(tfcc::runtime::common::FLOAT);
  std::vector<float> data = {1.0, 2.0};
  item->set_data(data.data(), data.size() * sizeof(float));

  model.run(inputs, outputs);

  return 0;
}