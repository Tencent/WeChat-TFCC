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

#include "tfcc_cudadeviceproperty.h"

#include <cuda_runtime.h>
#include <utility>

#include "exceptions/tfcc_cudaruntimeerror.h"

namespace tfcc {

CUDADeviceProperty::CUDADeviceProperty(unsigned deviceID) {
  cudaDeviceProp prop;
  cudaError_t ret = cudaGetDeviceProperties(&prop, deviceID);
  if (ret != cudaSuccess) throw CUDARuntimeError(ret);

  _maxThreadPerBlock = prop.maxThreadsPerBlock;
  _maxBlockPerGrimX = prop.maxGridSize[0];
  _maxBlockPerGrimY = prop.maxGridSize[1];
  _maxBlockPerGrimZ = prop.maxGridSize[2];
  _totalGlobalMemory = prop.totalGlobalMem;
}

std::tuple<size_t, size_t> CUDADeviceProperty::getSuitableKernelSize(size_t len) const {
  size_t blockCount = 1, threadCount = _maxThreadPerBlock;
  if (len < _maxThreadPerBlock) {
    threadCount = len;
  } else {
    blockCount = (len + _maxThreadPerBlock - 1) / _maxThreadPerBlock;
  }
  blockCount = std::min(blockCount, _maxBlockPerGrimX);
  return std::make_tuple(blockCount, threadCount);
}

std::tuple<size_t, size_t> CUDADeviceProperty::getSuitableKernelSize(
    size_t height, size_t width) const {
  size_t blockCount = height, threadCount = width;
  blockCount = std::min(blockCount, _maxBlockPerGrimX);
  threadCount = std::min(threadCount, _maxThreadPerBlock);
  return std::make_tuple(blockCount, threadCount);
}

std::string CUDADeviceProperty::toString() const {
  std::string str = "max threads per block: " + std::to_string(_maxThreadPerBlock) + "\n";
  str += "max blocks per grim: " + std::to_string(_maxBlockPerGrimX) + " " +
         std::to_string(_maxBlockPerGrimY) + " " + std::to_string(_maxBlockPerGrimZ) + "\n" +
         "total global memory: " + std::to_string(_totalGlobalMemory);
  return str;
}

size_t CUDADeviceProperty::getTotalGlobalMemory() const { return _totalGlobalMemory; }

}  // namespace tfcc
