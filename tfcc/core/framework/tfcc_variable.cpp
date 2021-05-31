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

#include "tfcc_variable.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_resourceexhaustederror.h"
#include "tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T>::Variable() : _session(Session::getThreadDefault()), _data(nullptr) {}

template <class T>
Variable<T>::Variable(const Shape& s)
    : Tensor<T>(s), _session(Session::getThreadDefault()), _data(nullptr) {
  if (this->Tensor<T>::_shape.area() > 0) {
    _data = reinterpret_cast<T*>(_session->malloc(sizeof(T) * this->Tensor<T>::_shape.area()));
    if (_data == nullptr) {
      throw ResourceExhaustedError();
    }
  }
}

template <class T>
Variable<T>::Variable(Shape&& s)
    : Tensor<T>(std::move(s)), _session(Session::getThreadDefault()), _data(nullptr) {
  if (this->Tensor<T>::_shape.area() > 0) {
    _data = reinterpret_cast<T*>(_session->malloc(sizeof(T) * this->Tensor<T>::_shape.area()));
    if (_data == nullptr) {
      throw ResourceExhaustedError();
    }
  }
}

template <class T>
Variable<T>::Variable(Variable&& v) noexcept
    : Tensor<T>(std::move(v)), _session(v._session), _data(v._data) {
  v._data = nullptr;
}

template <class T>
Variable<T>::~Variable() {
  _session->free(_data);
}

template <class T>
Variable<T>& Variable<T>::operator=(Variable&& v) noexcept {
  _session = v._session;
  this->Tensor<T>::operator=(std::move(v));
  _session->free(_data);
  _data = v._data;
  v._data = nullptr;
  return *this;
}

template <class T>
void Variable<T>::reshape(Shape s) {
  if (s.area() != this->Tensor<T>::_shape.area()) {
    throw InvalidArgumentError("shape area couldn't be changed");
  }
  this->Tensor<T>::_shape = std::move(s);
}

template <class T>
void Variable<T>::expandDims(size_t axis) {
  if (axis > this->Tensor<T>::_shape.size()) {
    throw InvalidArgumentError("invalid axis to expand");
  }
  std::vector<unsigned> s = this->Tensor<T>::_shape.toVector();
  s.insert(s.begin() + axis, 1);
  reshape(s);
}

template <class T>
void Variable<T>::squeeze() {
  std::vector<unsigned> s;
  for (size_t i = 0; i < this->Tensor<T>::_shape.size(); ++i) {
    if (this->Tensor<T>::_shape[i] != 1) {
      s.push_back(this->Tensor<T>::_shape[i]);
    }
  }
  reshape(s);
}

template <class T>
void Variable<T>::squeeze(const std::set<size_t>& axisSet) {
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

#define DEFINE_FUNC(type) template class Variable<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
template class Variable<Complex<float>>;
template class Variable<Complex<double>>;

}  // namespace tfcc
