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

#include "tfcc_random.h"

#include <chrono>
#include <random>

#include "framework/tfcc_types.h"
#include "interfaces/tfcc_activationinterface.h"
#include "interfaces/tfcc_arithmeticinterface.h"
#include "interfaces/tfcc_datainterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace random {

template <class T>
Variable<T> normal(Shape s, T mean, T stddev) {
  Variable<T> result(s);

  std::default_random_engine generator;
  std::normal_distribution<T> distribution(mean, stddev);
  std::vector<T> v(result.size());
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = distribution(generator);
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(result, v.data());
  return result;
}

template <class T>
Variable<T> logistics(Shape s, T minVal, T maxVal) {
  Variable<T> variableV(s), variableSubV(s);
  Variable<T> result(s);

  std::default_random_engine generator;

  std::uniform_real_distribution<T> distribution(minVal, maxVal);
  std::vector<T> v(result.size()), subv(result.size());
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = distribution(generator);
    subv[i] = 1.0 - v[i];
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(variableV, v.data());
  interface.getDataInterface().set(variableSubV, subv.data());

  variableV = interface.getActivationInterface().log(variableV);
  variableSubV = interface.getActivationInterface().log(variableSubV);
  result = interface.getArithmeticInterface().sub(variableV, variableSubV);
  return result;
}

template <class T>
Variable<T> binary(Shape s, T a, T b) {
  thread_local std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
  Variable<T> result(s);

  std::uniform_real_distribution<T> distribution(0, 1);
  std::vector<T> v(result.size());
  for (size_t i = 0; i < v.size(); ++i) {
    T x = distribution(generator);
    v[i] = x > 0.5 ? b : a;
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(result, v.data());
  return result;
}

template <class T>
Variable<T> uniform(Shape s, T minVal, T maxVal) {
  Variable<T> result(s);

  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(minVal, maxVal);
  std::vector<T> v(result.size());
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = distribution(generator);
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  interface.getDataInterface().set(result, v.data());
  return result;
}

#define DEFINE_FUNC(type)                                               \
  template Variable<type> normal(Shape s, type mean, type stddev);      \
  template Variable<type> logistics(Shape s, type minVal, type maxVal); \
  template Variable<type> binary(Shape s, type a, type b);              \
  template Variable<type> uniform(Shape s, type minVal, type maxVal);

TFCC_FOR_FLOATING_POINT_TYPES(DEFINE_FUNC);

}  // namespace random
}  // namespace tfcc
