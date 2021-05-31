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

#include "tfcc_base.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_basicinterface.h"
#include "interfaces/tfcc_datainterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace base {

template <class T>
Variable<T> slice(const Tensor<T>& a, size_t axis, unsigned start, unsigned end) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  if (start >= end) {
    throw InvalidArgumentError("start large than end");
  }
  if (start >= a.shape(axis)) {
    throw InvalidArgumentError("invalid start");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().slice(a, axis, start, end);
}

template <class T>
std::vector<Variable<T>> split(const Tensor<T>& a, size_t num, size_t axis) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  if (a.shape(axis) % num != 0) {
    throw InvalidArgumentError("invalid num");
  }
  unsigned s = a.shape(axis) / num;
  std::vector<Variable<T>> values;

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  for (size_t i = 0; i < num; ++i) {
    values.emplace_back(interface.getBasicInterface().slice(a, axis, i * s, (i + 1) * s));
  }
  return values;
}

template <class T>
std::vector<Variable<T>> split(
    const Tensor<T>& a, const std::vector<unsigned>& sizes, size_t axis) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  unsigned total = 0;
  for (unsigned s : sizes) {
    total += s;
  }
  if (total != a.shape(axis)) {
    throw InvalidArgumentError("invalid sizes");
  }
  unsigned start = 0;
  std::vector<Variable<T>> values;

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  for (unsigned s : sizes) {
    values.emplace_back(interface.getBasicInterface().slice(a, axis, start, start + s));
    start += s;
  }
  return values;
}

template <class T>
Variable<T> concat(const std::vector<const Tensor<T>*>& values, size_t axis) {
  if (values.size() == 0) {
    throw InvalidArgumentError("values.size could not be zero");
  }
  for (const Tensor<T>* tensor : values) {
    if (axis >= tensor->shape().size()) {
      throw InvalidArgumentError("invalid axis");
    }
    if (tensor->shape().size() != values[0]->shape().size()) {
      throw InvalidArgumentError("invalid shape");
    }
    for (size_t i = 0; i < tensor->shape().size(); ++i) {
      if (i != axis && tensor->shape(i) != values[0]->shape(i)) {
        throw InvalidArgumentError("invalid shape");
      }
    }
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().concat(values, axis);
}

template <class T>
Variable<T> concat(std::initializer_list<const Tensor<T>*> values, size_t axis) {
  std::vector<const Tensor<T>*> vs(values);
  return concat(vs, axis);
}

template <class T>
Variable<T> concat(const std::vector<Variable<T>>& values, size_t axis) {
  std::vector<const Tensor<T>*> vs;
  for (const auto& v : values) {
    vs.push_back(&v);
  }
  return concat(vs, axis);
}

template <class T>
Variable<T> transpose(const Tensor<T>& a, const std::vector<size_t>& perm) {
  if (a.shape().size() < 2 || a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (a.shape().size() != perm.size()) {
    throw InvalidArgumentError("the perm size is not equal to tensor's dimension");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().transpose(a, perm);
}

template <class T>
Variable<T> pad(const Tensor<T>& a, size_t axis, unsigned paddingHead, unsigned paddingEnd) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  std::vector<unsigned> s = a.shape().toVector();
  Variable<T> ph, pe;
  std::vector<const Tensor<T>*> concatList;
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  if (paddingHead > 0) {
    s[axis] = paddingHead;
    ph = Variable<T>(s);
    interface.getDataInterface().zeros(ph);
    concatList.push_back(&ph);
  }
  concatList.push_back(&a);
  if (paddingEnd > 0) {
    s[axis] = paddingEnd;
    pe = Variable<T>(s);
    interface.getDataInterface().zeros(pe);
    concatList.push_back(&pe);
  }

  return concat(concatList, axis);
}

template <class T>
void assign_to(const Tensor<T>& a, size_t axis, unsigned start, Variable<T>& b) {
  if (a.shape().size() != b.shape().size()) {
    throw InvalidArgumentError("dimensions not match");
  }
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  for (size_t i = 0; i < a.shape().size(); ++i) {
    if (i != axis && a.shape(i) != b.shape(i)) {
      throw InvalidArgumentError("shape not match");
    }
  }
  if (b.shape(axis) < start + a.shape(axis)) {
    throw InvalidArgumentError("b is too small to assign");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().assignTo(a, axis, start, b);
}

template <class T>
Variable<T> stack(const std::vector<const Tensor<T>*>& values, size_t axis) {
  if (values.size() == 0) {
    throw InvalidArgumentError("values's size should be greater than zero");
  }
  if (axis > values[0]->shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }

  std::vector<unsigned> s = values[0]->shape().toVector();
  s.insert(s.begin() + axis, 1);

  std::vector<View<T>> viewList;
  std::vector<const Tensor<T>*> tensorList;
  for (const Tensor<T>* tensor : values) {
    if (tensor->shape() != values[0]->shape()) {
      throw InvalidArgumentError("the shape of values are unequal");
    }
    View<T> view(*tensor, s);
    viewList.push_back(view);
  }

  for (View<T>& view : viewList) {
    tensorList.push_back(&view);
  }

  return concat(tensorList, axis);
}

template <class T>
Variable<T> stack(const std::initializer_list<const Tensor<T>*>& values, size_t axis) {
  std::vector<const Tensor<T>*> vs(values);
  return stack(vs, axis);
}

template <class T>
Variable<T> stack(const std::vector<Variable<T>>& values, size_t axis) {
  std::vector<const Tensor<T>*> vs;
  for (const auto& v : values) {
    vs.push_back(&v);
  }
  return stack(vs, axis);
}

template <class T>
std::vector<Variable<T>> unstack(const Tensor<T>& a, size_t axis) {
  if (a.shape().size() <= 1) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }

  auto s = a.shape().toVector();
  s.erase(s.begin() + axis);

  Shape shape(std::move(s));

  std::vector<Variable<T>> result = split(a, a.shape(axis), axis);
  for (Variable<T>& v : result) {
    v.reshape(shape);
  }
  return result;
}

template <class T>
Variable<T> tril(const Tensor<T>& a, int64_t k) {
  if (a.shape().size() != 2) {
    throw InvalidArgumentError("invalid shape");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().tril(a, k);
}

template <class T>
Variable<T> triu(const Tensor<T>& a, int64_t k) {
  if (a.shape().size() != 2) {
    throw InvalidArgumentError("invalid shape");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().triu(a, k);
}

#define DEFINE_FUNC(type)                                                                          \
  template Variable<type> slice(const Tensor<type>& a, size_t axis, unsigned start, unsigned end); \
  template std::vector<Variable<type>> split(const Tensor<type>& a, size_t num, size_t axis);      \
  template std::vector<Variable<type>> split(                                                      \
      const Tensor<type>& a, const std::vector<unsigned>& sizes, size_t axis);                     \
  template Variable<type> concat(const std::vector<const Tensor<type>*>& values, size_t axis);     \
  template Variable<type> concat(std::initializer_list<const Tensor<type>*> values, size_t axis);  \
  template Variable<type> concat(const std::vector<Variable<type>>& values, size_t axis);          \
  template Variable<type> pad(                                                                     \
      const Tensor<type>& a, size_t axis, unsigned paddingHead, unsigned paddingEnd);              \
  template void assign_to(const Tensor<type>& a, size_t axis, unsigned start, Variable<type>& b);  \
  template Variable<type> transpose(const Tensor<type>& a, const std::vector<size_t>& perm);       \
  template Variable<type> stack(const std::vector<const Tensor<type>*>& values, size_t axis);      \
  template Variable<type> stack(                                                                   \
      const std::initializer_list<const Tensor<type>*>& values, size_t axis);                      \
  template Variable<type> stack(const std::vector<Variable<type>>& values, size_t axis);           \
  template std::vector<Variable<type>> unstack(const Tensor<type>& a, size_t axis);                \
  template Variable<type> tril(const Tensor<type>& a, int64_t k);                                  \
  template Variable<type> triu(const Tensor<type>& a, int64_t k);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace base

}  // namespace tfcc
