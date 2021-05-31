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

#include "modeldataguard.h"

#include "tfcc.h"

#include "tfcc_runtime/exceptions/runtimeerror.h"

namespace tfcc {
namespace runtime {

static std::string _create_model_id() {
  static std::atomic<size_t> id(0);
  return "_tfcc_runtime_" + std::to_string(++id);
}

ModelDataGuard::ModelDataGuard(const std::string& data) : _mid(_create_model_id()) {
  tfcc::MultiDataLoader* multiLoader =
      dynamic_cast<tfcc::MultiDataLoader*>(tfcc::DataLoader::getGlobalDefault());
  if (multiLoader == nullptr) {
    throw RuntimeError("Invalid data loader status");
  }

  _loader = std::unique_ptr<tfcc::NPZDataLoader>(new tfcc::NPZDataLoader(data.data(), data.size()));
  multiLoader->addLoader(_mid, *_loader);
}

ModelDataGuard::~ModelDataGuard() {
  tfcc::Scope::getRootScope().removeChild(_mid);
  tfcc::MultiDataLoader* multiLoader =
      dynamic_cast<tfcc::MultiDataLoader*>(tfcc::DataLoader::getGlobalDefault());
  if (multiLoader != nullptr) {
    multiLoader->removeLoader(_mid);
  }
}

}  // namespace runtime
}  // namespace tfcc
