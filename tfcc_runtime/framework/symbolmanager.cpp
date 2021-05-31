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

#include "symbolmanager.h"

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tfcc_runtime/exceptions/runtimeerror.h"
#include "tfcc_runtime/exceptions/unknowsymbolerror.h"
#include "tfcc_runtime/framework/types.h"

namespace tfcc {
namespace runtime {

const SymbolInfo& SymbolManager::getSymbolInfo(const std::string& name) const {
  auto it = _symbolMap.find(name);
  if (it == _symbolMap.end()) {
    throw UnknowSymbolError(name);
  }
  return it->second;
}

template <class T>
tfcc::Tensor<T>* SymbolManager::getTensor(const std::string& name, T) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype == SymbolType::VARIABLE) {
    return &symbols[info.pos].variable;
  } else if (info.stype == SymbolType::VIEW) {
    return &symbols[info.pos].view;
  } else if (info.stype == SymbolType::CONSTANT_TENSOR) {
    return symbols[info.pos].constantTensor;
  } else {
    throw RuntimeError("invalid stype");
  }
}

template <class T>
tfcc::View<T>* SymbolManager::getView(const std::string& name, T) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype == SymbolType::VIEW) {
    return &symbols[info.pos].view;
  } else {
    throw RuntimeError("invalid stype");
  }
}

template <class T>
tfcc::Variable<T>* SymbolManager::getVariable(const std::string& name, T) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype == SymbolType::VARIABLE) {
    return &symbols[info.pos].variable;
  } else {
    throw RuntimeError("invalid stype");
  }
}

template <class T>
T* SymbolManager::getValue(const std::string& name, T) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype == SymbolType::VALUE) {
    return &symbols[info.pos].value;
  } else if (info.stype == SymbolType::CONSTANT_VALUE) {
    return &symbols[info.pos].constantValue;
  } else {
    throw RuntimeError("invalid stype");
  }
}

template <class T>
std::vector<T>* SymbolManager::getVector(const std::string& name, T) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype == SymbolType::VECTOR) {
    return &symbols[info.pos].vector;
  } else if (info.stype == SymbolType::CONSTANT_VECTOR) {
    return &symbols[info.pos].constantVector;
  } else {
    throw RuntimeError("invalid stype");
  }
}

template <class T>
void SymbolManager::setView(const std::string& name, tfcc::Variable<T>&& variable) {
  const SymbolInfo& info = getSymbolInfo(name);
  std::vector<Symbol<T>>& symbols = getSymbols(info.dtype, T());
  if (info.pos >= symbols.size()) {
    throw RuntimeError("invalid pos");
  }
  if (info.stype != SymbolType::VIEW) {
    throw RuntimeError("invalid stype");
  }
  symbols[info.pos].variable = std::move(variable);
  symbols[info.pos].view = tfcc::View<T>(symbols[info.pos].variable);
}

void SymbolManager::createSymbol(const tfcc::runtime::model::Symbol& symbol) {
  if (symbol.data_type() == tfcc::runtime::common::FLOAT) {
    createSymbol(symbol, _floatSymbols);
  } else if (symbol.data_type() == tfcc::runtime::common::DOUBLE) {
    createSymbol(symbol, _doubleSymbols);
  } else if (symbol.data_type() == tfcc::runtime::common::UINT8) {
    createSymbol(symbol, _uint8Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::INT8) {
    createSymbol(symbol, _int8Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::UINT16) {
    createSymbol(symbol, _uint16Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::INT16) {
    createSymbol(symbol, _int16Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::UINT32) {
    createSymbol(symbol, _uint32Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::INT32) {
    createSymbol(symbol, _int32Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::UINT64) {
    createSymbol(symbol, _uint64Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::INT64) {
    createSymbol(symbol, _int64Symbols);
  } else if (symbol.data_type() == tfcc::runtime::common::BOOL) {
    createSymbol(symbol, _boolSymbols);
  } else if (symbol.data_type() == tfcc::runtime::common::COMPLEX64) {
    createSymbol(symbol, _complex64Symbols);
  } else {
    throw RuntimeError("Invalid data type " + std::to_string(symbol.data_type()));
  }
}

template <class T>
void SymbolManager::createSymbol(
    const tfcc::runtime::model::Symbol& symbol, std::vector<Symbol<T>>& target) {
  if (_symbolMap.find(symbol.name()) != _symbolMap.end()) {
    throw RuntimeError("name " + symbol.name() + " has existed");
  }

  SymbolInfo info;
  info.dtype = symbol.data_type();
  if (symbol.has_variable()) {
    info.stype = SymbolType::VARIABLE;
  } else if (symbol.has_view()) {
    info.stype = SymbolType::VIEW;
  } else if (symbol.has_constant_tensor()) {
    info.stype = SymbolType::CONSTANT_TENSOR;
  } else if (symbol.has_value()) {
    info.stype = SymbolType::VALUE;
  } else if (symbol.has_constant_value()) {
    info.stype = SymbolType::CONSTANT_VALUE;
  } else if (symbol.has_vector()) {
    info.stype = SymbolType::VECTOR;
  } else if (symbol.has_constant_vector()) {
    info.stype = SymbolType::CONSTANT_VECTOR;
  } else {
    throw RuntimeError("invalid stype");
  }
  target.emplace_back();
  info.pos = target.size() - 1;

  // set constant
  if (info.stype == SymbolType::CONSTANT_TENSOR) {
    target[info.pos].constantTensor =
        &tfcc::Constant<T>::getConstant(symbol.constant_tensor().ref());
  } else if (info.stype == SymbolType::CONSTANT_VALUE) {
    target[info.pos].constantValue =
        tfcc::Configure<T>::getConfigure(symbol.constant_value().ref());
  } else if (info.stype == SymbolType::CONSTANT_VECTOR) {
    target[info.pos].constantVector =
        tfcc::data::get(tfcc::Constant<T>::getConstant(symbol.constant_vector().ref()));
  }

  _symbolMap[symbol.name()] = info;
}

void SymbolManager::createSymbol(
    const tfcc::runtime::model::Symbol& symbol, std::vector<Symbol<tfcc::Complex<float>>>& target) {
  if (_symbolMap.find(symbol.name()) != _symbolMap.end()) {
    throw RuntimeError("name " + symbol.name() + " has existed");
  }

  SymbolInfo info;
  info.dtype = symbol.data_type();
  if (symbol.has_variable()) {
    info.stype = SymbolType::VARIABLE;
  } else if (symbol.has_view()) {
    info.stype = SymbolType::VIEW;
  } else if (symbol.has_value()) {
    info.stype = SymbolType::VALUE;
  } else if (symbol.has_vector()) {
    info.stype = SymbolType::VECTOR;
  } else {
    throw RuntimeError("invalid stype");
  }
  target.emplace_back();
  info.pos = target.size() - 1;

  _symbolMap[symbol.name()] = info;
}

template <>
std::vector<Symbol<float>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, float) {
  if (dtype != tfcc::runtime::common::FLOAT) {
    throw RuntimeError("type not match");
  }
  return _floatSymbols;
}

template <>
std::vector<Symbol<double>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, double) {
  if (dtype != tfcc::runtime::common::DOUBLE) {
    throw RuntimeError("type not match");
  }
  return _doubleSymbols;
}

template <>
std::vector<Symbol<uint8_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, uint8_t) {
  if (dtype != tfcc::runtime::common::UINT8 && dtype != tfcc::runtime::common::BOOL) {
    throw RuntimeError("type not match");
  }
  if (dtype == tfcc::runtime::common::UINT8) {
    return _uint8Symbols;
  } else {
    return _boolSymbols;
  }
}

template <>
std::vector<Symbol<int8_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, int8_t) {
  if (dtype != tfcc::runtime::common::INT8) {
    throw RuntimeError("type not match");
  }
  return _int8Symbols;
}

template <>
std::vector<Symbol<uint16_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, uint16_t) {
  if (dtype != tfcc::runtime::common::UINT16) {
    throw RuntimeError("type not match");
  }
  return _uint16Symbols;
}

template <>
std::vector<Symbol<int16_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, int16_t) {
  if (dtype != tfcc::runtime::common::INT16) {
    throw RuntimeError("type not match");
  }
  return _int16Symbols;
}

template <>
std::vector<Symbol<uint32_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, uint32_t) {
  if (dtype != tfcc::runtime::common::UINT32) {
    throw RuntimeError("type not match");
  }
  return _uint32Symbols;
}

template <>
std::vector<Symbol<int32_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, int32_t) {
  if (dtype != tfcc::runtime::common::INT32) {
    throw RuntimeError("type not match");
  }
  return _int32Symbols;
}

template <>
std::vector<Symbol<uint64_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, uint64_t) {
  if (dtype != tfcc::runtime::common::UINT64) {
    throw RuntimeError("type not match");
  }
  return _uint64Symbols;
}

template <>
std::vector<Symbol<int64_t>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, int64_t) {
  if (dtype != tfcc::runtime::common::INT64) {
    throw RuntimeError("type not match");
  }
  return _int64Symbols;
}

template <>
std::vector<Symbol<tfcc::Complex<float>>>& SymbolManager::getSymbols(
    tfcc::runtime::common::DataType dtype, tfcc::Complex<float>) {
  if (dtype != tfcc::runtime::common::COMPLEX64) {
    throw RuntimeError("type not match");
  }
  return _complex64Symbols;
}

#define DEFINE_FUNC(type)                                                                   \
  template tfcc::Tensor<type>* SymbolManager::getTensor(const std::string& name, type);     \
  template tfcc::View<type>* SymbolManager::getView(const std::string& name, type);         \
  template tfcc::Variable<type>* SymbolManager::getVariable(const std::string& name, type); \
  template type* SymbolManager::getValue(const std::string& name, type);                    \
  template std::vector<type>* SymbolManager::getVector(const std::string& name, type);      \
  template void SymbolManager::setView(const std::string& name, tfcc::Variable<type>&& variable);

TFCC_RUNTIME_FOR_ALL_TYPES(DEFINE_FUNC);
TFCC_RUNTIME_FOR_COMPLEX_TYPES(DEFINE_FUNC);

}  // namespace runtime
}  // namespace tfcc
