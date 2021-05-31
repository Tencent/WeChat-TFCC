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

#include "tfcc_mklsession.h"

#include <future>
#include <utility>

#include <dnnl.hpp>

#include "allocators/tfcc_flexallocator.h"
#include "framework/tfcc_mkldevice.h"

namespace tfcc {

MKLSession::MKLSession(MKLDevice& device)
    : MKLSession(device, std::unique_ptr<Allocator>(new FlexAllocator())) {}

MKLSession::MKLSession(MKLDevice& device, std::unique_ptr<Allocator> allocator)
    : Session(std::move(allocator)),
      _device(device),
      _currentTaskID(0),
      _syncTaskID(0),
      _spinWaitTimes(0),
      _exception(nullptr),
      _dnnlStream(nullptr),
      _instructionFlags(device.getCPUInstructionFlags()) {
  _allocator->setRealMalloc([this](size_t len) { return this->_device.malloc(len); });

  _allocator->setRealFree([this](void* p) {
    this->sync();
    this->_device.free(p);
  });

  _dnnlStream = new dnnl::stream(device.getDNNLEngine());
}

MKLSession::~MKLSession() {
  preRelease();
  delete _dnnlStream;
}

void MKLSession::sync() const {
  waitSync();
  std::exception_ptr e = _exception;
  _exception = nullptr;
  if (e) {
    std::rethrow_exception(e);
  }
}

Device& MKLSession::getDevice() { return _device; }

const Device& MKLSession::getDevice() const { return _device; }

void MKLSession::addTask(const std::function<void()>& func, std::string taskName) {
  size_t id = ++_currentTaskID;
  _device.addTask(
      [this, id, func]() {
        if (!this->_exception) {
          try {
            func();
          } catch (std::exception& e) {
            _exception = std::current_exception();
          }
        }
        this->_syncTaskID.store(id);
      },
      std::move(taskName));
}

void MKLSession::setSpinWaitTimes(long spinWaitTimes) { _spinWaitTimes = spinWaitTimes; }

void MKLSession::waitSync() const {
  if (_spinWaitTimes < 0) {
    while (_syncTaskID.load() != _currentTaskID) {
      continue;
    }
    return;
  }

  for (long i = 0; i < _spinWaitTimes && _syncTaskID.load() != _currentTaskID; ++i) {
    continue;
  }
  if (_syncTaskID.load() == _currentTaskID) {
    return;
  }

  std::promise<void> promise;
  std::future<void> future = promise.get_future();
  _device.addTask(
      [this, &promise]() {
        if (this->_exception) {
          promise.set_exception(this->_exception);
          this->_exception = nullptr;
        } else {
          promise.set_value();
        }
      },
      "_tfcc_session_sync");
  future.get();
}

const dnnl::engine& MKLSession::getDNNLEngine() const { return _device.getDNNLEngine(); }

void MKLSession::setCPUInstructionFlags(uint64_t flags) {
  _instructionFlags = flags & _device.getCPUInstructionFlags();
}

}  // namespace tfcc
