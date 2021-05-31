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

#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>

namespace tfcc {

template <class T>
FIFOQueue<T>::FIFOQueue(size_t capacity)
    : _len(capacity + 1),
      _data(nullptr),
      _status(new std::atomic<bool>[_len]),
      _pushStart(0),
      _pushClaim(0),
      _popStart(0),
      _popClaim(0),
      _pushWaitCount(0),
      _popWaitCount(0) {
  _data = reinterpret_cast<T*>(new char[sizeof(T) * _len]);
  memset(_status.get(), 0, _len * sizeof(bool));
}

template <class T>
FIFOQueue<T>::~FIFOQueue() {
  while (!empty()) tryPop();
}

template <class T>
inline void FIFOQueue<T>::push(const T& v) {
  size_t pos = allocPushClaim();

  new (_data + pos) T(v);
  assert(_status[pos].load() == false);
  _status[pos].store(true);

  finishPush();
}

template <class T>
inline void FIFOQueue<T>::push(T&& v) {
  size_t pos = allocPushClaim();

  new (_data + pos) T(std::move(v));
  assert(_status[pos].load() == false);
  _status[pos].store(true);

  finishPush();
}

template <class T>
inline bool FIFOQueue<T>::tryPush(const T& v) {
  size_t pos = tryAllocPushClaim();
  if (pos == _len) return false;

  new (_data + pos) T(v);
  assert(_status[pos].load() == false);
  _status[pos].store(true);

  finishPush();
  return true;
}

template <class T>
inline bool FIFOQueue<T>::tryPush(T&& v) {
  size_t pos = tryAllocPushClaim();
  if (pos == _len) return false;

  new (_data + pos) T(std::move(v));
  assert(_status[pos].load() == false);
  _status[pos].store(true);

  finishPush();
  return true;
}

template <class T>
inline T FIFOQueue<T>::pop() {
  size_t pos = allocPopClaim();

  T v(std::move(_data[pos]));
  _data[pos].~T();
  assert(_status[pos].load() == true);
  _status[pos].store(false);

  finishPop();
  return v;
}

template <class T>
inline bool FIFOQueue<T>::tryPop(T& v) {
  size_t pos = tryAllocPopClaim();
  if (pos == _len) return false;

  v = std::move(_data[pos]);
  _data[pos].~T();
  assert(_status[pos].load() == true);
  _status[pos].store(false);

  finishPop();
  return true;
}

template <class T>
inline bool FIFOQueue<T>::tryPop() {
  size_t pos = tryAllocPopClaim();
  if (pos == _len) return false;

  T v(std::move(_data[pos]));
  _data[pos].~T();
  assert(_status[pos].load() == true);
  _status[pos].store(false);

  finishPop();
  return true;
}

template <class T>
inline bool FIFOQueue<T>::empty() {
  return _popClaim.load() == _pushStart.load();
}

template <class T>
inline bool FIFOQueue<T>::full() {
  return (_pushClaim.load() + 1) % _len == _popStart.load();
}

template <class T>
inline size_t FIFOQueue<T>::allocPushClaim() {
  size_t pos;
  while (true) {
    pos = _pushClaim.load();
    size_t nextPos = (pos + 1) % _len;
    if (nextPos == _popStart) {
      ++_pushWaitCount;
      if (nextPos == _popStart) {
        std::unique_lock<std::mutex> lck(_pushMutex);
        _pushCV.wait_for(lck, std::chrono::milliseconds(1));
      }
      --_pushWaitCount;
      continue;
    }

    if (_pushClaim.compare_exchange_weak(pos, nextPos)) break;
  }

  return pos;
}

template <class T>
inline size_t FIFOQueue<T>::tryAllocPushClaim() {
  size_t pos;
  while (true) {
    pos = _pushClaim.load();
    size_t nextPos = (pos + 1) % _len;
    if (nextPos == _popStart) return _len;

    if (_pushClaim.compare_exchange_weak(pos, nextPos)) break;
  }

  return pos;
}

template <class T>
inline void FIFOQueue<T>::finishPush() {
  while (true) {
    size_t pos = _pushStart.load();
    size_t nextPos = (pos + 1) % _len;
    if (!_status[pos].load() || pos == _pushClaim) break;
    _pushStart.compare_exchange_weak(pos, nextPos);
  }

  if (_popWaitCount.load() > 0 && !empty()) _popCV.notify_all();
}

template <class T>
inline size_t FIFOQueue<T>::allocPopClaim() {
  size_t pos;
  while (true) {
    pos = _popClaim.load();
    size_t nextPos = (pos + 1) % _len;
    if (pos == _pushStart) {
      ++_popWaitCount;
      if (pos == _pushStart) {
        std::unique_lock<std::mutex> lck(_popMutex);
        _popCV.wait_for(lck, std::chrono::milliseconds(1));
      }
      --_popWaitCount;
      continue;
    }

    if (_popClaim.compare_exchange_weak(pos, nextPos)) break;
  }
  return pos;
}

template <class T>
inline size_t FIFOQueue<T>::tryAllocPopClaim() {
  size_t pos;
  while (true) {
    pos = _popClaim.load();
    size_t nextPos = (pos + 1) % _len;
    if (pos == _pushStart) return _len;

    if (_popClaim.compare_exchange_weak(pos, nextPos)) break;
  }
  return pos;
}

template <class T>
inline void FIFOQueue<T>::finishPop() {
  while (true) {
    size_t pos = _popStart.load();
    size_t nextPos = (pos + 1) % _len;
    if (_status[pos].load() || pos == _popClaim) break;
    _popStart.compare_exchange_weak(pos, nextPos);
  }

  if (_pushWaitCount.load() > 0 && !full()) _pushCV.notify_all();
}

}  // namespace tfcc
