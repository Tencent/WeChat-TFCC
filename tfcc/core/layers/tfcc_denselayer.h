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

#include <string>

#include "framework/tfcc_constant.h"
#include "framework/tfcc_tensor.h"
#include "framework/tfcc_variable.h"

namespace tfcc {
namespace layer {

template <class T>
class Dense {
  bool _initialized;
  Constant<T>* _kernel;
  Constant<T>* _bias;
  unsigned _units;
  bool _useBias;
  std::string _name;
  std::string _kernelName;
  std::string _biasName;

 public:
  /**
   * units:   Dimensionality of the output space. If zero, units will be initialized by auto.
   * useBias: Whether the layer uses a bias.
   * name:    Name of the layer. The default value is 'dense'
   */
  Dense(unsigned units, bool useBias);
  Dense(unsigned units, bool useBias, std::string name);

  Variable<T> operator()(const Tensor<T>& inputs);

  void setKernelName(std::string name);
  void setBiasName(std::string name);
};

template <class T>
Variable<T> dense(const Tensor<T>& inputs, unsigned units, bool useBias);

template <class T>
Variable<T> dense(const Tensor<T>& inputs, unsigned units, bool useBias, std::string name);

template <class T>
Variable<T> dense(
    const Tensor<T>& inputs, unsigned units, bool useBias, std::string name, std::string kernelName,
    std::string biasName);

}  // namespace layer
}  // namespace tfcc
