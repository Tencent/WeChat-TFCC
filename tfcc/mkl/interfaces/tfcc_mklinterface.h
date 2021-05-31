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

#include "framework/tfcc_types.h"
#include "interfaces/tfcc_interface.h"

namespace tfcc {

class MKLDevice;

template <class T>
Interface<T> get_mkl_interface(T, const MKLDevice& device);

template <class T>
Interface<Complex<T>> get_mkl_complex_interface(Complex<T>, const MKLDevice& device);

}  // namespace tfcc
