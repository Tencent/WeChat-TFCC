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

#include "tfcc_blas.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_blasinterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace blas {

template <class T>
Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.shape().size() < 2 || b.shape().size() < 2) {
    throw InvalidArgumentError("matmul require tensors with two or more dimensions");
  }
  if (a.shape().size() != b.shape().size() && a.shape().size() != 2 && b.shape().size() != 2) {
    throw InvalidArgumentError("shape invalid");
  }
  if (a.shape(a.shape().size() - 1) != b.shape(b.shape().size() - 2)) {
    throw InvalidArgumentError("shape invalid");
  }
  if (a.shape().size() == b.shape().size()) {
    for (size_t i = 2; i < a.shape().size(); ++i) {
      if (a.shape(i - 2) != b.shape(i - 2)) {
        throw InvalidArgumentError("shape invalid");
      }
    }
  }

  if (a.size() == 0 || b.size() == 0) {
    unsigned m = a.shape(a.shape().size() - 2);
    unsigned n = b.shape(b.shape().size() - 1);
    std::vector<unsigned> resultS =
        a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
    resultS[resultS.size() - 2] = m;
    resultS[resultS.size() - 1] = n;

    Variable<T> result(std::move(resultS));
    return result;
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBlasInterface().matmul(a, b);
}

template <class T>
Variable<T> matmul(const Tensor<T>& a, const Tensor<T>& b, const Tensor<T>& c) {
  if (a.shape().size() < 2 || b.shape().size() < 2) {
    throw InvalidArgumentError("matmul require tensors with two or more dimensions");
  }
  if (a.shape().size() != b.shape().size() && a.shape().size() != 2 && b.shape().size() != 2) {
    throw InvalidArgumentError("shape invalid");
  }
  if (a.shape(a.shape().size() - 1) != b.shape(b.shape().size() - 2)) {
    throw InvalidArgumentError("shape invalid");
  }
  if (c.shape().size() != 1 || c.size() != b.shape(b.shape().size() - 1)) {
    throw InvalidArgumentError("shape invalid");
  }
  if (a.shape().size() == b.shape().size()) {
    for (size_t i = 2; i < a.shape().size(); ++i) {
      if (a.shape(i - 2) != b.shape(i - 2)) {
        throw InvalidArgumentError("shape invalid");
      }
    }
  }

  if (a.size() == 0 || b.size() == 0) {
    unsigned m = a.shape(a.shape().size() - 2);
    unsigned n = b.shape(b.shape().size() - 1);
    std::vector<unsigned> resultS =
        a.shape().size() > b.shape().size() ? a.shape().toVector() : b.shape().toVector();
    resultS[resultS.size() - 2] = m;
    resultS[resultS.size() - 1] = n;

    Variable<T> result(std::move(resultS));
    return result;
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBlasInterface().matmul(a, b, c);
}

#define DEFINE_FUNC(type)                                                       \
  template Variable<type> matmul(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<type> matmul(                                               \
      const Tensor<type>& a, const Tensor<type>& b, const Tensor<type>& c);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace blas
}  // namespace tfcc
