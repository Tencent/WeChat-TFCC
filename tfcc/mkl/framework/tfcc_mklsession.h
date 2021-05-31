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

#include <atomic>
#include <exception>

#include "framework/tfcc_session.h"

namespace dnnl {
class stream;
class engine;
}  // namespace dnnl

namespace tfcc {

class MKLDevice;

class MKLSession : public Session {
  MKLDevice& _device;
  size_t _currentTaskID;
  std::atomic<size_t> _syncTaskID;
  long _spinWaitTimes;
  mutable std::exception_ptr _exception;
  dnnl::stream* _dnnlStream;
  uint64_t _instructionFlags;

 public:
  explicit MKLSession(MKLDevice& device);
  MKLSession(MKLDevice&, std::unique_ptr<Allocator> allocator);
  MKLSession(MKLSession&&) = default;
  ~MKLSession();

  void sync() const override;
  Device& getDevice() override;
  const Device& getDevice() const override;

  /**
   * Add task to dispath stream.
   * @param func A function or callable object to dispatch.
   * @param taskName The name of function or callable object.
   */
  void addTask(const std::function<void()>& func, std::string taskName);

  /**
   * Set Spin wait times on sync.
   * @param spinWaitTimes Spin wait times on sync. If -1, always spin on sync.
   */
  void setSpinWaitTimes(long spinWaitTimes);

  /**
   * Get DNNL stream
   * @return reference of stream.
   */
  dnnl::stream& getDNNLStream() { return *_dnnlStream; }

  /**
   * Get DNNL engine
   * @return Const reference of engine.
   */
  const dnnl::engine& getDNNLEngine() const;

  /**
   * Set which instruction will be used in this session.
   * @param flags Instruction flags.
   * @see MKLCPUInstruction
   */
  void setCPUInstructionFlags(uint64_t flags);

  /**
   * Get current instruction in using.
   * @return Instruction flags.
   */
  uint64_t getCPUInstructionFlags() const { return _instructionFlags; }

 private:
  void waitSync() const;
};

}  // namespace tfcc
