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

#include "tfcc_mkldevice.h"

#include <omp.h>
#include <xmmintrin.h>
#include <atomic>
#include <cstdlib>
#include <dnnl.hpp>
#include <utility>

#include "allocators/tfcc_flexallocator.h"
#include "framework/tfcc_constantmanager.h"
#include "framework/tfcc_scope.h"
#include "interfaces/tfcc_mklinterface.h"
#include "utils/tfcc_coster.h"
#include "utils/tfcc_spinlock.h"

#include "framework/tfcc_mklinstruction.h"

namespace tfcc {

template <class T>
static std::shared_ptr<ConstantManager<T>> _get_constant_manager(T x) {
  static std::weak_ptr<ConstantManager<T>> constantManager;
  static SpinLock mtx;

  auto result = constantManager.lock();
  if (result) {
    return result;
  }
  std::lock_guard<SpinLock> lck(mtx);
  result = constantManager.lock();
  if (result) {
    return result;
  }
  result = std::make_shared<ConstantManager<T>>();
  result->getAllocator().setRealMalloc(
      [](size_t len) -> void* { return ::aligned_alloc(64, len); });

  result->getAllocator().setRealFree([](void* p) { ::free(p); });
  constantManager = result;
  return result;
}

MKLDevice::MKLDevice() : MKLDevice(std::unique_ptr<Allocator>(new FlexAllocator())) {}

MKLDevice::MKLDevice(std::unique_ptr<Allocator> allocator)
    : Device(allocDeviceID(), std::move(allocator)),
      _taskQueue(1024 * 1024),
      _dnnlEngine(nullptr),
      _instructionFlags(get_cpu_instruction_set()) {
  _allocator->setRealMalloc([](size_t len) -> void* { return ::aligned_alloc(64, len); });

  _allocator->setRealFree([](void* p) { ::free(p); });

  _dispatchThread.reset(new std::thread([this]() { this->dispatchLoop(); }));
  _dnnlEngine = new dnnl::engine(dnnl::engine::kind::cpu, 0);

  // set constant manager
  _floatConstantManager = _get_constant_manager(float());
  _doubleConstantManager = _get_constant_manager(double());
  _int8ConstantManager = _get_constant_manager(int8_t());
  _uint8ConstantManager = _get_constant_manager(uint8_t());
  _int16ConstantManager = _get_constant_manager(int16_t());
  _uint16ConstantManager = _get_constant_manager(uint16_t());
  _int32ConstantManager = _get_constant_manager(int32_t());
  _uint32ConstantManager = _get_constant_manager(uint32_t());
  _int64ConstantManager = _get_constant_manager(int64_t());
  _uint64ConstantManager = _get_constant_manager(uint64_t());

  // set interface
  _floatInterface = get_mkl_interface(float(), *this);
  _doubleInterface = get_mkl_interface(double(), *this);
  _int8Interface = get_mkl_interface(int8_t(), *this);
  _uint8Interface = get_mkl_interface(uint8_t(), *this);
  _int16Interface = get_mkl_interface(int16_t(), *this);
  _uint16Interface = get_mkl_interface(uint16_t(), *this);
  _int32Interface = get_mkl_interface(int32_t(), *this);
  _uint32Interface = get_mkl_interface(uint32_t(), *this);
  _int64Interface = get_mkl_interface(int64_t(), *this);
  _uint64Interface = get_mkl_interface(uint64_t(), *this);
  _complex64Interface = get_mkl_complex_interface(Complex<float>{}, *this);
  _complex128Interface = get_mkl_complex_interface(Complex<double>{}, *this);
}

MKLDevice::~MKLDevice() {
  _taskQueue.push(_MKLDeviceTaskInfo());
  _dispatchThread->join();
  delete _dnnlEngine;
}

void MKLDevice::attach() {}

void MKLDevice::addTask(std::function<void()> func, std::string taskName) {
  if (func) {
    _MKLDeviceTaskInfo taskInfo;
    taskInfo.func = std::move(func);
    taskInfo.taskName = std::move(taskName);
    _taskQueue.push(std::move(taskInfo));
  }
}

void MKLDevice::setNumberThread(size_t num) {
  addTask([num]() { omp_set_num_threads(num); }, "_tfcc_set_num_threads");
}

const MKLTaskStatistics& MKLDevice::getStatistics() const { return _statistics; }

void MKLDevice::clearStatistics() {
  addTask([this]() { this->_statistics.clear(); }, "");
}

void MKLDevice::dispatchLoop() {
  _mm_setcsr(_mm_getcsr() | 0x0040);
  _mm_setcsr(_mm_getcsr() | 0x8000);
  while (true) {
    Coster coster;
    _MKLDeviceTaskInfo taskInfo = _taskQueue.pop();
    if (!taskInfo.func) {
      break;
    }
    _statistics.statistics("_wait_async_queue", coster.lap().microseconds());

    coster.reset();
    taskInfo.func();
    std::string name = std::move(taskInfo.taskName);
    _statistics.statistics(name, coster.lap().microseconds());
  }
}

unsigned MKLDevice::allocDeviceID() noexcept {
  static std::atomic<unsigned> lastAllocNumber(0);

  unsigned deviceID = static_cast<unsigned>(DeviceType::MKL) << 24;
  deviceID += lastAllocNumber++;

  return deviceID;
}

}  // namespace tfcc
