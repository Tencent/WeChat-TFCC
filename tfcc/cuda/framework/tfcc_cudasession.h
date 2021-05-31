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

#include <memory>

#ifdef WITH_CUDA_HEADER
#  include <cublas_v2.h>
#  include <cuda_runtime.h>
#  include <cudnn.h>
#endif

#include "allocators/tfcc_allocator.h"
#include "framework/tfcc_session.h"

namespace tfcc {

class CUDASessionImpl;
#ifdef WITH_CUDA_HEADER
class CUDASessionImpl {
  cudaStream_t _cudaStream;
  cublasHandle_t _cublasHandle;
  cudnnHandle_t _cudnnHandle;

  friend class CUDASession;

 public:
  cudaStream_t cudaStream() const { return _cudaStream; }

  cublasHandle_t cublasHandle() const { return _cublasHandle; }

  cudnnHandle_t cudnnHandle() const { return _cudnnHandle; }
};
#endif

class CUDADevice;

class CUDASession : public Session {
  CUDADevice& _device;
  std::unique_ptr<CUDASessionImpl> _cudaSessionImpl;
  bool _destroying;

 public:
  explicit CUDASession(CUDADevice& device);
  CUDASession(CUDADevice& device, std::unique_ptr<Allocator> allocator);
  CUDASession(CUDASession&& session) = default;
  ~CUDASession();

  void sync() const override;
  Device& getDevice() override;
  const Device& getDevice() const override;

  const CUDASessionImpl* getImpl() const { return _cudaSessionImpl.get(); }

 private:
  void initImpl(CUDASessionImpl* impl);
};

}  // namespace tfcc
