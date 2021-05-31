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

#include <cuda_runtime.h>

#include "exceptions/tfcc_exception.h"

namespace tfcc {

class CUDARuntimeError : public Exception {
  cudaError_t _status;

 public:
  explicit CUDARuntimeError(cudaError_t status);

  const char* reason() const noexcept override;
};

}  // namespace tfcc
