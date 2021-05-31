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

#include "environment.h"

#include <stdexcept>

#include "tfcc.h"

#ifdef TFCC_WITH_MKL
#  include "framework/tfcc_mkldevice.h"
#  include "framework/tfcc_mklsession.h"
#endif

#ifdef TFCC_WITH_CUDA
#  include "framework/tfcc_cudadevice.h"
#  include "framework/tfcc_cudasession.h"
#endif

tfcc::DeviceType Environment::_defaultDeviceType = tfcc::DeviceType::CUDA;
tfcc::DataLoader* Environment::_defaultTestDataLoader = nullptr;
std::string Environment::_defaultTestDataPath;

void Environment::init() {
  _currentDeviceType = _defaultDeviceType;
#ifdef TFCC_WITH_MKL
  if (_defaultDeviceType == tfcc::DeviceType::MKL) {
    tfcc::MKLDevice* device = new tfcc::MKLDevice();
    device->setNumberThread(4);
    tfcc::MKLSession* session = new tfcc::MKLSession(*device);
    _device = device;
    _session = session;
  }
#endif
#ifdef TFCC_WITH_CUDA
  if (_defaultDeviceType == tfcc::DeviceType::CUDA) {
    tfcc::CUDADevice* device = new tfcc::CUDADevice(0);
    tfcc::CUDASession* session = new tfcc::CUDASession(*device);
    _device = device;
    _session = session;
  }
#endif
  if (_device == nullptr || _session == nullptr) {
    throw std::runtime_error("unknow device type");
  }

  tfcc::DataLoader::setGlobalDefault(_defaultTestDataLoader);
  tfcc::Device::setThreadDefault(_device);
  tfcc::Session::setThreadDefault(_session);
}

void Environment::release() {
  tfcc::Session::setThreadDefault(nullptr);
  tfcc::Device::setThreadDefault(nullptr);
  tfcc::DataLoader::setGlobalDefault(nullptr);

  delete _session;
  _session = nullptr;

  delete _device;
  _device = nullptr;
}

void Environment::setDefaultTestDataPath(const std::string& path) {
  _defaultTestDataPath = path;
  _defaultTestDataLoader = new tfcc::NPZDataLoader(path + "/tfcc_test_data.npz");
}

std::string Environment::getDefaultTestDataPath() { return _defaultTestDataPath; }

std::vector<size_t> revert_transpose_perm(std::vector<size_t> perm) {
  std::vector<size_t> newPerm = perm;
  for (size_t i = 0; i < perm.size(); ++i) {
    newPerm[perm[i]] = i;
  }
  return newPerm;
}
