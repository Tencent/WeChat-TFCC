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

#include "framework/tfcc_shape.h"
#include "framework/tfcc_tensor.h"

namespace tfcc {

template <class T>
class View : public Tensor<T> {
  const Tensor<T>* _tensor;
  size_t _offset;

 public:
  View();
  /**
   * view of tensor
   */
  explicit View(const Tensor<T>& tensor);
  /**
   * view of tensor which reshape to s
   */
  View(const Tensor<T>& tensor, const Shape& s);
  /**
   * view of tensor which reshape to s and slice to `tensor[start:]`
   */
  View(const Tensor<T>& tensor, const Shape& s, unsigned start);
  /**
   * view of tensor which reshape to s and slice to `tensor[start:end]`
   */
  View(const Tensor<T>& tensor, const Shape& s, unsigned start, unsigned end);

  View(const View& view);
  View(View&& view) noexcept = default;

  ~View();

  View& operator=(const View& view);
  View& operator=(View&& view) noexcept = default;

  /**
   * @warning Don't Use this function directly! Please use tfcc::data::get instead, If you want get
   * the value of the tensor.
   */
  const T* data() const override { return _tensor->data() + _offset; }

  void sync() const override { _tensor->sync(); }

  bool available() const override { return _tensor->available(); }

  /**
   * Reshapes the view.
   * @param s New shape of the view.
   */
  void reshape(Shape s);

  /**
   * Inserts a dimension of 1 into the view's shape.
   * @param axis The dimension index to insert.
   */
  void expandDims(size_t axis);

  /**
   * Remove all size 1 dimensions.
   */
  void squeeze();

  /**
   * Remove specific size 1 dimensions.
   * @param axisSet The dimension index set.
   */
  void squeeze(const std::set<size_t>& axisSet);
};

}  // namespace tfcc
