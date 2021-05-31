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

#include <functional>
#include <memory>
#include <ostream>

#include "framework/tfcc_device.h"
#include "utils/tfcc_fifoqueue.h"
#include "utils/tfcc_mkltaskstatistics.h"

namespace dnnl {
class engine;
}

namespace tfcc {

struct _MKLDeviceTaskInfo {
  std::function<void()> func;
  std::string taskName;
};

class MKLDevice : public Device {
  std::unique_ptr<std::thread> _dispatchThread;
  FIFOQueue<_MKLDeviceTaskInfo> _taskQueue;
  MKLTaskStatistics _statistics;
  dnnl::engine* _dnnlEngine;
  uint64_t _instructionFlags;

 public:
  MKLDevice();
  explicit MKLDevice(std::unique_ptr<Allocator> allocator);
  MKLDevice(MKLDevice&&) = default;
  ~MKLDevice();

  MKLDevice& operator=(const MKLDevice&) = delete;
  MKLDevice& operator=(MKLDevice&&) = delete;

  void attach() override;

  /**
   * Add task to dispath stream.
   * @param func A function or callable object to dispatch.
   * @param taskName The name of function or callable object.
   */
  void addTask(std::function<void()> func, std::string taskName);

  /**
   * Specifies the number of OpenMP threads for all Intel MKL functions and openmp functions on the
   * current execution thread.
   * @param num The number of threads.
   */
  void setNumberThread(size_t num);

  /**
   * Get task statistics.
   * @return The task statistics reference.
   * @deprecated
   */
  const MKLTaskStatistics& getStatistics() const;

  /**
   * Clear task statistics
   */
  void clearStatistics();

  /**
   * Get DNNL engine
   * @return Const reference of engine.
   */
  const dnnl::engine& getDNNLEngine() const { return *_dnnlEngine; }

  /**
   * Get instruction of current CPU.
   * @return Instruction flags.
   */
  uint64_t getCPUInstructionFlags() const { return _instructionFlags; }

 private:
  void dispatchLoop();

 private:
  static unsigned allocDeviceID() noexcept;
};

}  // namespace tfcc
