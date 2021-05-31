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

#include "tfcc_view.h"

#include <algorithm>
#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "tfcc_types.h"

namespace tfcc {

template <class T>
View<T>::View() : _tensor(nullptr), _offset(0) {}

template <class T>
View<T>::View(const View& view)
    : Tensor<T>(view._shape), _tensor(view._tensor), _offset(view._offset) {}

template <class T>
View<T>::View(const Tensor<T>& tensor) : View(tensor, tensor.shape(), 0, tensor.shape(0)) {}

template <class T>
View<T>::View(const Tensor<T>& tensor, const Shape& s) : View(tensor, s, 0, s[0]) {}

template <class T>
View<T>::View(const Tensor<T>& tensor, const Shape& s, unsigned start)
    : View(tensor, s, start, s[0]) {}

template <class T>
View<T>::View(const Tensor<T>& tensor, const Shape& s, unsigned start, unsigned end)
    : _tensor(&tensor) {
  if (s.area() != tensor.size() || s.size() == 0) {
    throw InvalidArgumentError("invalid shape");
  }
  if (start >= s[0] && s[0] > 0) {
    throw InvalidArgumentError("invalid start");
  }
  end = std::min(end, s[0]);
  std::vector<unsigned> newS = s.toVector();
  newS[0] = end - start;
  this->Tensor<T>::_shape = Shape(std::move(newS));
  if (start > 0) {
    _offset = s.area() / s[0];
    _offset *= start;
  } else {
    _offset = 0;
  }
}

template <class T>
View<T>::~View() {}

template <class T>
View<T>& View<T>::operator=(const View& view) {
  this->Tensor<T>::_shape = view._shape;
  _tensor = view._tensor;
  _offset = view._offset;
  return *this;
}

template <class T>
void View<T>::reshape(Shape s) {
  if (s.area() != this->Tensor<T>::_shape.area()) {
    throw InvalidArgumentError("shape area couldn't be changed");
  }
  this->Tensor<T>::_shape = std::move(s);
}

template <class T>
void View<T>::expandDims(size_t axis) {
  if (axis > this->Tensor<T>::_shape.size()) {
    throw InvalidArgumentError("invalid axis to expand");
  }
  std::vector<unsigned> s = this->Tensor<T>::_shape.toVector();
  s.insert(s.begin() + axis, 1);
  reshape(s);
}

template <class T>
void View<T>::squeeze() {
  std::vector<unsigned> s;
  for (size_t i = 0; i < this->Tensor<T>::_shape.size(); ++i) {
    if (this->Tensor<T>::_shape[i] != 1) {
      s.push_back(this->Tensor<T>::_shape[i]);
    }
  }
  reshape(s);
}

template <class T>
void View<T>::squeeze(const std::set<size_t>& axisSet) {
  if (*axisSet.rbegin() >= this->Tensor<T>::_shape.size()) {
    throw InvalidArgumentError("invalid axis to squeeze");
  }

  std::vector<unsigned> s;
  for (size_t i = 0; i < this->Tensor<T>::_shape.size(); ++i) {
    if (axisSet.find(i) != axisSet.end()) {
      if (this->Tensor<T>::_shape[i] != 1) {
        throw InvalidArgumentError("invalid axis to squeeze");
      }
      continue;
    }
    s.push_back(this->Tensor<T>::_shape[i]);
  }
  reshape(s);
}

#define DEFINE_FUNC(type) template class View<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
template class View<Complex<float>>;
template class View<Complex<double>>;

}  // namespace tfcc
