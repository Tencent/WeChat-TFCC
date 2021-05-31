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

#include "tfcc_cublasruntimeerror.h"

namespace tfcc {

CUBlasRuntimeError::CUBlasRuntimeError(cublasStatus_t status) : _status(status) {}

const char* CUBlasRuntimeError::reason() const noexcept {
  switch (_status) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "The cuBLAS library was not initialized.";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "Resource allocation failed inside the cuBLAS library.";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "An unsupported value or parameter was passed to the function (a negative vector "
             "size, for example).";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "The function requires a feature absent from the device architecture; usually caused "
             "by the lack of support for double precision.";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "An access to GPU memory space failed, which is usually caused by a failure to bind a "
             "texture.";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "An internal cuBLAS operation failed.";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "The functionnality requested is not supported.";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "The functionnality requested requires some license and an error was detected when "
             "trying to check the current licensing.";
    default:
      return "Unknow error.";
  }
}

}  // namespace tfcc
