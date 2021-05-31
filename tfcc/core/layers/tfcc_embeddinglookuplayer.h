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

#pragma once

#include <string>
#include <vector>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {

namespace layer {

enum class EmbeddingPartitionType {
  NONE = 0,
  MOD = 1,
  DIV = 2,
};

template <class T>
class EmbeddingLookup {
  bool _initialized;
  std::vector<Constant<T>*> _embeddings;
  std::string _name;
  EmbeddingPartitionType _type;
  size_t _partitionCount;
  size_t _totalSize;

 public:
  explicit EmbeddingLookup(std::string name);
  EmbeddingLookup(std::string name, EmbeddingPartitionType type, size_t partitionCount);

  Variable<T> operator()(const std::vector<unsigned>& ids);
  Variable<T> operator()(const std::vector<unsigned>& ids, const Shape& idShape);
};

}  // namespace layer

}  // namespace tfcc
