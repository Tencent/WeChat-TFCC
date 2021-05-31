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

#include "interfaces/tfcc_gatherinterface.h"

namespace tfcc {

template <class T>
class MKLGatherInterface : public GatherInterface<T> {
 public:
  Variable<T> gather(const Tensor<T>& params, const Tensor<uint32_t>& indices) override;
  Variable<T> gather(const Tensor<T>& params, const Tensor<int32_t>& indices) override;
  Variable<T> gather(const Tensor<T>& params, const Tensor<uint64_t>& indices) override;
  Variable<T> gather(const Tensor<T>& params, const Tensor<int64_t>& indices) override;
};

};  // namespace tfcc
