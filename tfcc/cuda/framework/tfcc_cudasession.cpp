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

#include "tfcc_cudasession.h"

#include <utility>

#include "allocators/tfcc_flexallocator.h"
#include "exceptions/tfcc_cublasruntimeerror.h"
#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_cudnnruntimeerror.h"
#include "tfcc_cudadevice.h"

namespace tfcc {

CUDASession::CUDASession(CUDADevice& device)
    : CUDASession(device, std::unique_ptr<Allocator>(new FlexAllocator())) {
  FlexAllocator* alloc = dynamic_cast<FlexAllocator*>(_allocator.get());
  if (alloc) {
    size_t totalGlobalMemory = _device.getProperty().getTotalGlobalMemory();
    alloc->setLimit(totalGlobalMemory * 6 / 10);
  }
}

CUDASession::CUDASession(CUDADevice& device, std::unique_ptr<Allocator> allocator)
    : Session(std::move(allocator)),
      _device(device),
      _cudaSessionImpl(nullptr),
      _destroying(false) {
  std::unique_ptr<CUDASessionImpl> impl(new CUDASessionImpl);
  cudaError_t ret = cudaStreamCreate(&impl->_cudaStream);
  if (ret != 0) throw CUDARuntimeError(ret);

  cublasStatus_t cublas_ret = cublasCreate(&impl->_cublasHandle);
  if (cublas_ret != CUBLAS_STATUS_SUCCESS) throw CUBlasRuntimeError(cublas_ret);
  cublas_ret = cublasSetStream(impl->_cublasHandle, impl->_cudaStream);
  if (cublas_ret != CUBLAS_STATUS_SUCCESS) throw CUBlasRuntimeError(cublas_ret);

  cudnnStatus_t cudnn_ret = cudnnCreate(&impl->_cudnnHandle);
  if (cudnn_ret != CUDNN_STATUS_SUCCESS) throw CUDNNRuntimeError(cudnn_ret);
  cudnn_ret = cudnnSetStream(impl->_cudnnHandle, impl->_cudaStream);
  if (cudnn_ret != CUDNN_STATUS_SUCCESS) throw CUDNNRuntimeError(cudnn_ret);

  _cudaSessionImpl = std::move(impl);

  _allocator->setRealMalloc([this](size_t len) { return this->_device.malloc(len); });

  _allocator->setRealFree([this](void* p) {
    this->sync();
    this->_device.free(p);
  });
}

CUDASession::~CUDASession() {
  _destroying = true;
  preRelease();
  cudnnDestroy(_cudaSessionImpl->_cudnnHandle);
  cublasDestroy(_cudaSessionImpl->_cublasHandle);
  cudaStreamDestroy(_cudaSessionImpl->_cudaStream);
}

void CUDASession::sync() const {
  cudaError_t ret = cudaStreamSynchronize(_cudaSessionImpl->_cudaStream);
  if (ret != cudaSuccess) {
    if (ret == cudaErrorCudartUnloading && _destroying) return;
    throw CUDARuntimeError(ret);
  }
}

Device& CUDASession::getDevice() { return _device; }

const Device& CUDASession::getDevice() const { return _device; }

}  // namespace tfcc
