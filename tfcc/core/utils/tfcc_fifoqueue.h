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
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>

namespace tfcc {

/**
 * A lock-free thread safe FIFO queue.
 */
template <class T>
class FIFOQueue {
  const size_t _len;
  T* _data;
  std::unique_ptr<std::atomic<bool>[]> _status;
  std::atomic<size_t> _pushStart, _pushClaim, _popStart, _popClaim;
  std::atomic<size_t> _pushWaitCount, _popWaitCount;
  std::mutex _pushMutex, _popMutex;
  std::condition_variable _pushCV, _popCV;

 public:
  /**
   * @param capacity The capacity of the queue.
   */
  explicit FIFOQueue(size_t capacity);
  ~FIFOQueue();

  /**
   * Inserts a new element at the end of the queue. The function may block if necessary until a free
   * slot is available.
   * @param v The element to insert.
   */
  void push(const T& v);

  /**
   * @see push(const T&)
   */
  void push(T&& v);

  /**
   * Inserts a new element at the end of the queue if a free slot is immediately available, else do
   * nothing.
   * @param v The element to insert.
   * @return true if this operation is successful.
   */
  bool tryPush(const T& v);

  /**
   * @see tryPush(const T&)
   */
  bool tryPush(T&& v);

  /**
   * Get the next element in the queue. The function may block if necessary until a element is
   * avalable.
   * @return The next element in the queue.
   */
  T pop();

  /**
   * Get the next element in the queue if a element is immediately available, else do nothing.
   * @param v A reference to save the next element in the queue.
   * @return true if this operation is successful.
   */
  bool tryPop(T& v);

  /**
   * Remove the next element in the queue if a element is immediately available, else do nothing.
   * @return true if this operation is successful.
   */
  bool tryPop();

  /**
   * @return true if the queue is empty.
   */
  bool empty();

  /**
   * @return true if the queue is full.
   */
  bool full();

 private:
  size_t allocPushClaim();
  size_t tryAllocPushClaim();
  void finishPush();

  size_t allocPopClaim();
  size_t tryAllocPopClaim();
  void finishPop();
};

}  // namespace tfcc

#include "tfcc_fifoqueue.hpp"
