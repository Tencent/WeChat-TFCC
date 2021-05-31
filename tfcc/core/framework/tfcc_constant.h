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
#include <string>

#include "framework/tfcc_shape.h"
#include "framework/tfcc_tensor.h"
#include "utils/tfcc_spinlock.h"

namespace tfcc {

template <class T>
class ConstantManager;

class Allocator;

template <class T>
class Constant : public Tensor<T> {
  T* _data;
  Allocator& _allocator;
  SpinLock _mtx;

 private:
  explicit Constant(Allocator& allocator);

  void setData(const Shape& shape, const std::vector<T>& data);

  friend class ConstantManager<T>;

 public:
  Constant(const Constant&) = delete;
  Constant(Constant&&) = delete;
  ~Constant();

  Constant& operator=(const Constant&) = delete;
  Constant& operator=(Constant&&) = delete;

  /**
   * @warning Don't Use this function directly! Please use tfcc::data::get instead, If you want get
   * the value of the tensor.
   */
  const T* data() const override { return _data; }

  void sync() const override {}

  bool available() const override;

  typedef std::function<std::pair<Shape, std::vector<T>>(Shape, std::vector<T>)> ProcFunc;
  static Constant& getConstant(const std::string& name, ProcFunc preprocessFunc);
  static Constant& getConstant(const std::string& name) { return getConstant(name, nullptr); }
  static Constant& getConstantWithTranspose(
      const std::string& name, const std::vector<size_t>& perm);

  static Constant* tryGetRuntimeConstant(const std::string& name);
  /**
   * @return True if success.
   */
  static bool setRuntimeConstantIfNotExist(
      const std::string& name, Shape shape, const std::vector<T>& data);
};

}  // namespace tfcc
