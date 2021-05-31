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
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "tfcc.h"
#include "tfcc_runtime/proto/common.pb.h"
#include "tfcc_runtime/proto/model.pb.h"

namespace tfcc {
namespace runtime {

enum class SymbolType {
  VARIABLE = 1,
  VIEW = 2,
  CONSTANT_TENSOR = 3,
  VALUE = 4,
  CONSTANT_VALUE = 5,
  VECTOR = 6,
  CONSTANT_VECTOR = 7,
};

inline bool is_tensor(SymbolType stype) {
  return stype == SymbolType::VARIABLE || stype == SymbolType::VIEW ||
         stype == SymbolType::CONSTANT_TENSOR;
}

inline bool is_value(SymbolType stype) {
  return stype == SymbolType::VALUE || stype == SymbolType::CONSTANT_VALUE;
}

inline bool is_vector(SymbolType stype) {
  return stype == SymbolType::VECTOR || stype == SymbolType::CONSTANT_VECTOR;
}

inline bool is_integer(tfcc::runtime::common::DataType dtype) {
  if (dtype == tfcc::runtime::common::UINT8 || dtype == tfcc::runtime::common::INT8) {
    return true;
  }
  if (dtype == tfcc::runtime::common::UINT16 || dtype == tfcc::runtime::common::INT16) {
    return true;
  }
  if (dtype == tfcc::runtime::common::UINT32 || dtype == tfcc::runtime::common::INT32) {
    return true;
  }
  if (dtype == tfcc::runtime::common::UINT64 || dtype == tfcc::runtime::common::INT64) {
    return true;
  }
  return false;
}

inline bool is_floating_point(tfcc::runtime::common::DataType dtype) {
  return dtype == tfcc::runtime::common::FLOAT || dtype == tfcc::runtime::common::DOUBLE;
}

inline bool is_signed(tfcc::runtime::common::DataType dtype) {
  if (dtype == tfcc::runtime::common::FLOAT || dtype == tfcc::runtime::common::DOUBLE) {
    return true;
  }
  if (dtype == tfcc::runtime::common::INT8) {
    return true;
  }
  if (dtype == tfcc::runtime::common::INT16) {
    return true;
  }
  if (dtype == tfcc::runtime::common::INT32) {
    return true;
  }
  if (dtype == tfcc::runtime::common::INT64) {
    return true;
  }
  return false;
}

struct SymbolInfo {
  tfcc::runtime::common::DataType dtype;
  SymbolType stype;
  size_t pos = std::numeric_limits<size_t>::max();
};

template <class T>
struct Symbol {
  tfcc::Variable<T> variable;
  tfcc::View<T> view;
  tfcc::Constant<T>* constantTensor;
  T value;
  T constantValue;
  std::vector<T> vector;
  std::vector<T> constantVector;
};

class SymbolManager {
  std::unordered_map<std::string, SymbolInfo> _symbolMap;

  std::vector<Symbol<float>> _floatSymbols;
  std::vector<Symbol<double>> _doubleSymbols;
  std::vector<Symbol<uint8_t>> _uint8Symbols;
  std::vector<Symbol<int8_t>> _int8Symbols;
  std::vector<Symbol<uint16_t>> _uint16Symbols;
  std::vector<Symbol<int16_t>> _int16Symbols;
  std::vector<Symbol<uint32_t>> _uint32Symbols;
  std::vector<Symbol<int32_t>> _int32Symbols;
  std::vector<Symbol<uint64_t>> _uint64Symbols;
  std::vector<Symbol<int64_t>> _int64Symbols;
  std::vector<Symbol<uint8_t>> _boolSymbols;
  std::vector<Symbol<tfcc::Complex<float>>> _complex64Symbols;
  std::vector<Symbol<tfcc::Complex<double>>> _complex128Symbols;

 public:
  template <class Iter>
  SymbolManager(Iter begin, Iter end) {
    for (auto it = begin; it != end; ++it) {
      createSymbol(*it);
    }
  }

  const SymbolInfo& getSymbolInfo(const std::string& name) const;

  template <class T>
  tfcc::Tensor<T>* getTensor(const std::string& name, T);
  template <class T>
  tfcc::View<T>* getView(const std::string& name, T);
  template <class T>
  tfcc::Variable<T>* getVariable(const std::string& name, T);
  template <class T>
  T* getValue(const std::string& name, T);
  template <class T>
  std::vector<T>* getVector(const std::string& name, T);
  template <class T>
  void setView(const std::string& name, tfcc::Variable<T>&& variable);

 private:
  void createSymbol(const tfcc::runtime::model::Symbol& symbol);
  template <class T>
  void createSymbol(const tfcc::runtime::model::Symbol& symbol, std::vector<Symbol<T>>& target);

  void createSymbol(
      const tfcc::runtime::model::Symbol& symbol,
      std::vector<Symbol<tfcc::Complex<float>>>& target);

  template <class T>
  std::vector<Symbol<T>>& getSymbols(tfcc::runtime::common::DataType dtype, T);
};

}  // namespace runtime
}  // namespace tfcc
