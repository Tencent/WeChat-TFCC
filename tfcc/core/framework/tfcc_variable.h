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

#include <set>

#include "framework/tfcc_session.h"
#include "framework/tfcc_tensor.h"

namespace tfcc {

template <class T>
class Variable : public Tensor<T> {
  Session* _session;
  T* _data;

 public:
  Variable();
  explicit Variable(const Shape& s);
  explicit Variable(Shape&& s);
  Variable(const Variable&) = delete;
  Variable(Variable&& s) noexcept;
  ~Variable();

  Variable& operator=(const Variable&) = delete;
  Variable& operator=(Variable&&) noexcept;

  /**
   * @warning Don't Use this function directly! Please use tfcc::data::get instead, If you want get
   * the value of the tensor.
   */
  const T* data() const override { return _data; }

  /**
   * @warning Don't Use this function directly! Please use tfcc::data::get instead, If you want get
   * the value of the tensor.
   */
  T* data() { return _data; }

  void sync() const override { _session->sync(); }

  bool available() const override { return _session == Session::getThreadDefault(); }

  /**
   * Reshapes the variable.
   * @param s New shape of the variable.
   */
  void reshape(Shape s);

  /**
   * Inserts a dimension of 1 into the variable's shape.
   * @param axis The dimension index to insert.
   */
  void expandDims(size_t axis);

  /**
   * Remove all size 1 dimensions.
   */
  void squeeze();

  /**
   * Remove specific size 1 dimensions.
   * @param axisSet The dimension index list.
   */
  void squeeze(const std::set<size_t>& axisSet);
};

}  // namespace tfcc
