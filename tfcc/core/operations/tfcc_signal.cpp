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

#include "tfcc_signal.h"

#include <chrono>
#include <random>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_signalinterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace signal {

template <class T>
Variable<Complex<T>> rfft(const Tensor<T>& a, unsigned length) {
  if (a.size() == 0 || length == 0) {
    throw InvalidArgumentError();
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getSignalInterface().rfft(a, length);
}

template <class T>
Variable<T> irfft(const Tensor<Complex<T>>& a, unsigned length) {
  if (a.size() == 0 || length == 0) {
    throw InvalidArgumentError();
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getSignalInterface().irfft(a, length);
}

#define DEFINE_FUNC(type)                                                        \
  template Variable<Complex<type>> rfft(const Tensor<type>& a, unsigned length); \
  template Variable<type> irfft(const Tensor<Complex<type>>& a, unsigned length);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace signal
}  // namespace tfcc
