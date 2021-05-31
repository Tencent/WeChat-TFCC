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

#include "tfcc_normalizationinterface.h"

#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_types.h"

namespace tfcc {

template <class T>
Variable<T> NormalizationInterface<T>::layerNormalize(
    const Tensor<T>& a, const Tensor<T>& gamma, const Tensor<T>& beta, T epsilon,
    size_t beginNormAxis) {
  throw NotImplementedError();
}

template <class T>
Variable<T> NormalizationInterface<T>::localResponseNormalization(
    const Tensor<T>& a, T alpha, T beta, T bias, unsigned size) {
  throw NotImplementedError();
}

template <class T>
Variable<T> NormalizationInterface<T>::batchNormalization(
    const Tensor<T>& a, size_t axis, const Tensor<T>& scale, const Tensor<T>& offset,
    const Tensor<T>& mean, const Tensor<T>& var, T epsilon) {
  throw NotImplementedError();
}

#define DEFINE_FUNC(type) template class NormalizationInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace tfcc
