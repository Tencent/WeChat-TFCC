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

#include <cstdint>

#include "tfcc.h"
#include "tfcc_runtime/proto/common.pb.h"

namespace tfcc {
namespace runtime {

#define TFCC_RUNTIME_FOR_ALL_TYPES(macro) \
  macro(float);                           \
  macro(double);                          \
  macro(int8_t);                          \
  macro(uint8_t);                         \
  macro(int16_t);                         \
  macro(uint16_t);                        \
  macro(int32_t);                         \
  macro(uint32_t);                        \
  macro(int64_t);                         \
  macro(uint64_t)

#define TFCC_RUNTIME_FOR_COMPLEX_TYPES(macro) macro(tfcc::Complex<float>);

#define TFCC_RUNTIME_FOR_ALL_DATA_TYPE(macro) \
  macro(tfcc::runtime::common::FLOAT);        \
  macro(tfcc::runtime::common::DOUBLE);       \
  macro(tfcc::runtime::common::UINT8);        \
  macro(tfcc::runtime::common::INT8);         \
  macro(tfcc::runtime::common::UINT16);       \
  macro(tfcc::runtime::common::INT16);        \
  macro(tfcc::runtime::common::UINT32);       \
  macro(tfcc::runtime::common::INT32);        \
  macro(tfcc::runtime::common::UINT64);       \
  macro(tfcc::runtime::common::INT64);        \
  macro(tfcc::runtime::common::BOOL);

#define TFCC_RUNTIME_FOR_COMPLEX_DATA_TYPE(macro) macro(tfcc::runtime::common::COMPLEX64);

#define TFCC_RUNTIME_FOR_FLOATING_POINT_DATA_TYPES(macro) \
  macro(tfcc::runtime::common::FLOAT);                    \
  macro(tfcc::runtime::common::DOUBLE);

#define _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, type) \
  macro(type, tfcc::runtime::common::FLOAT);                 \
  macro(type, tfcc::runtime::common::DOUBLE);                \
  macro(type, tfcc::runtime::common::UINT8);                 \
  macro(type, tfcc::runtime::common::INT8);                  \
  macro(type, tfcc::runtime::common::UINT16);                \
  macro(type, tfcc::runtime::common::INT16);                 \
  macro(type, tfcc::runtime::common::UINT32);                \
  macro(type, tfcc::runtime::common::INT32);                 \
  macro(type, tfcc::runtime::common::UINT64);                \
  macro(type, tfcc::runtime::common::INT64);                 \
  macro(type, tfcc::runtime::common::BOOL);

#define TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2(macro)                               \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::FLOAT);  \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::DOUBLE); \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::UINT8);  \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::INT8);   \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::UINT16); \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::INT16);  \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::UINT32); \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::INT32);  \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::UINT64); \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::INT64);  \
  _TFCC_RUNTIME_FOR_ALL_DATA_TYPE_ARG2_L1(macro, tfcc::runtime::common::BOOL);

template <tfcc::runtime::common::DataType dtype>
struct DataTypeInfo {};

template <>
struct DataTypeInfo<tfcc::runtime::common::FLOAT> {
  using CPPType = float;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::DOUBLE> {
  using CPPType = double;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::UINT8> {
  using CPPType = uint8_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::INT8> {
  using CPPType = int8_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::UINT16> {
  using CPPType = uint16_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::INT16> {
  using CPPType = int16_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::UINT32> {
  using CPPType = uint32_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::INT32> {
  using CPPType = int32_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::UINT64> {
  using CPPType = uint64_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::INT64> {
  using CPPType = int64_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::BOOL> {
  using CPPType = uint8_t;
};

template <>
struct DataTypeInfo<tfcc::runtime::common::COMPLEX64> {
  using CPPType = tfcc::Complex<float>;
};

}  // namespace runtime
}  // namespace tfcc
