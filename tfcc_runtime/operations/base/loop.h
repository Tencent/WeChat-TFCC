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
#include <memory>
#include <vector>

#include "tfcc.h"

#include "tfcc_runtime/framework/symbolmanager.h"
#include "tfcc_runtime/framework/types.h"
#include "tfcc_runtime/operations/operation.h"
#include "tfcc_runtime/proto/common.pb.h"

namespace tfcc {
namespace runtime {
namespace base {

template <class T>
struct ScanValue {
  T value;
  tfcc::Variable<T> variable;
  std::vector<T> vector;
};

struct ScanCollection {
  std::vector<ScanValue<float>> floatValues;
  std::vector<ScanValue<double>> doubleValues;
  std::vector<ScanValue<uint8_t>> uint8Values;
  std::vector<ScanValue<int8_t>> int8Values;
  std::vector<ScanValue<uint16_t>> uint16Values;
  std::vector<ScanValue<int16_t>> int16Values;
  std::vector<ScanValue<uint32_t>> uint32Values;
  std::vector<ScanValue<int32_t>> int32Values;
  std::vector<ScanValue<uint64_t>> uint64Values;
  std::vector<ScanValue<int64_t>> int64Values;
  std::vector<ScanValue<uint8_t>> boolValues;
};

struct LoopResource : public OperationResource {
  uint64_t maxLoop;
  uint64_t currentLoop;
  std::vector<ScanCollection> scans;
};

class Loop : public Operation {
 public:
  std::string getOperationTypeUrl() const override;
  std::set<unsigned> getOperationVersions() const override;
  std::unique_ptr<OperationResource> process(
      const tfcc::runtime::model::Node& node, Graph& graph) const override;

 private:
  void setIteratorSymbol(
      tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
      const uint64_t* it) const;
  void appendScanValue(
      tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
      ScanCollection& collection) const;
  template <class T>
  void appendScanValueInner(
      tfcc::Xbyak::CodeGenerator& jit, SymbolManager& manager, std::string name,
      std::vector<ScanValue<T>>& values) const;
  void getScanResult(
      tfcc::Xbyak::CodeGenerator& jit, SymbolType scanStype, SymbolManager& manager,
      std::string name, ScanCollection& collection) const;
  template <class T>
  void getScanResultInner(
      tfcc::Xbyak::CodeGenerator& jit, SymbolType scanStype, SymbolManager& manager,
      std::string name, std::vector<ScanValue<T>>& values) const;
};

std::vector<std::unique_ptr<Operation>> get_loop_operations();

}  // namespace base
}  // namespace runtime
}  // namespace tfcc
