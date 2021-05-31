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

#include "tfcc_cudadevice.h"

#include <cuda_runtime.h>
#include <utility>

#include "allocators/tfcc_flexallocator.h"
#include "exceptions/tfcc_cudaruntimeerror.h"
#include "exceptions/tfcc_resourceexhaustederror.h"
#include "framework/tfcc_constantmanager.h"
#include "framework/tfcc_scope.h"
#include "interfaces/tfcc_cudainterface.h"
#include "tfcc_cudadeviceproperty.h"

namespace tfcc {

CUDADevice::CUDADevice(unsigned deviceID)
    : CUDADevice(deviceID, std::unique_ptr<Allocator>(new FlexAllocator())) {
  FlexAllocator* alloc = dynamic_cast<FlexAllocator*>(_allocator.get());
  if (alloc) alloc->setLimit(_property.getTotalGlobalMemory() * 9 / 10);
}

CUDADevice::CUDADevice(unsigned deviceID, std::unique_ptr<Allocator> allocator)
    : Device(deviceID & 0x00ffffff, std::move(allocator)),
      _property(deviceID & 0x00ffffff),
      _tensorCore(true) {
  auto realMalloc = [](size_t len) -> void* {
    void* result = nullptr;
    cudaError_t ret = cudaMalloc(&result, len);
    if (ret != cudaSuccess) return nullptr;
    return result;
  };
  auto realFree = [](void* p) {
    cudaError_t ret = cudaFree(p);
    if (ret != cudaSuccess) {
      if (ret == cudaErrorCudartUnloading) return;
      throw CUDARuntimeError(ret);
    }
  };
  _allocator->setRealMalloc(realMalloc);

  _allocator->setRealFree(realFree);

  // set constant manager
  _floatConstantManager = std::make_shared<ConstantManager<float>>();
  _floatConstantManager->getAllocator().setRealMalloc(realMalloc);
  _floatConstantManager->getAllocator().setRealFree(realFree);

  _doubleConstantManager = std::make_shared<ConstantManager<double>>();
  _doubleConstantManager->getAllocator().setRealMalloc(realMalloc);
  _doubleConstantManager->getAllocator().setRealFree(realFree);

  _int8ConstantManager = std::make_shared<ConstantManager<int8_t>>();
  _int8ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _int8ConstantManager->getAllocator().setRealFree(realFree);

  _uint8ConstantManager = std::make_shared<ConstantManager<uint8_t>>();
  _uint8ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _uint8ConstantManager->getAllocator().setRealFree(realFree);

  _int16ConstantManager = std::make_shared<ConstantManager<int16_t>>();
  _int16ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _int16ConstantManager->getAllocator().setRealFree(realFree);

  _uint16ConstantManager = std::make_shared<ConstantManager<uint16_t>>();
  _uint16ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _uint16ConstantManager->getAllocator().setRealFree(realFree);

  _int32ConstantManager = std::make_shared<ConstantManager<int32_t>>();
  _int32ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _int32ConstantManager->getAllocator().setRealFree(realFree);

  _uint32ConstantManager = std::make_shared<ConstantManager<uint32_t>>();
  _uint32ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _uint32ConstantManager->getAllocator().setRealFree(realFree);

  _int64ConstantManager = std::make_shared<ConstantManager<int64_t>>();
  _int64ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _int64ConstantManager->getAllocator().setRealFree(realFree);

  _uint64ConstantManager = std::make_shared<ConstantManager<uint64_t>>();
  _uint64ConstantManager->getAllocator().setRealMalloc(realMalloc);
  _uint64ConstantManager->getAllocator().setRealFree(realFree);

  // set interfaces
  _floatInterface = get_cuda_interface(float(), _property);
  _doubleInterface = get_cuda_interface(double(), _property);
  _int8Interface = get_cuda_interface(int8_t(), _property);
  _uint8Interface = get_cuda_interface(uint8_t(), _property);
  _int16Interface = get_cuda_interface(int16_t(), _property);
  _uint16Interface = get_cuda_interface(uint16_t(), _property);
  _int32Interface = get_cuda_interface(int32_t(), _property);
  _uint32Interface = get_cuda_interface(uint32_t(), _property);
  _int64Interface = get_cuda_interface(int64_t(), _property);
  _uint64Interface = get_cuda_interface(uint64_t(), _property);
}

CUDADevice::~CUDADevice() {}

void CUDADevice::attach() {
  cudaError_t ret = cudaSetDevice(_deviceID);
  if (ret != cudaSuccess) throw CUDARuntimeError(ret);
}

const CUDADeviceProperty& CUDADevice::getProperty() const { return _property; }

size_t CUDADevice::getDeviceCount() {
  int count = 0;
  cudaError_t ret = cudaGetDeviceCount(&count);
  if (ret != cudaSuccess) throw CUDARuntimeError(ret);
  count = count < 0 ? 0 : count;
  return count;
}

void CUDADevice::enableTensorCore() { _tensorCore = true; }

void CUDADevice::disableTensorCore() { _tensorCore = false; }

bool CUDADevice::isTensorCoreEnabled() const { return _tensorCore; }

}  // namespace tfcc
