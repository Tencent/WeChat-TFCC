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

#include "tfcc_constant.h"

#include <cstdint>
#include <mutex>
#include <utility>

#include "allocators/tfcc_allocator.h"
#include "dataloaders/tfcc_dataloader.h"
#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_resourceexhaustederror.h"
#include "exceptions/tfcc_runtimeerror.h"
#include "framework/tfcc_constantmanager.h"
#include "framework/tfcc_device.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_datainterface.h"
#include "operations/tfcc_data.h"
#include "utils/tfcc_commutils.h"

namespace tfcc {

template <class T>
Constant<T>::Constant(Allocator& allocator) : _data(nullptr), _allocator(allocator) {}

template <class T>
Constant<T>::~Constant() {
  if (_data) {
    _allocator.free(_data);
  }
}

template <class T>
bool Constant<T>::available() const {
  Device* device = Device::getThreadDefault();
  if (device == nullptr) {
    return false;
  }
  return &device->getConstantManager(T()).getAllocator() == &_allocator;
}

template <class T>
void Constant<T>::setData(const Shape& shape, const std::vector<T>& data) {
  if (_data != nullptr) {
    throw RuntimeError("Reset constant data");
  }
  this->Tensor<T>::_shape = shape;
  _data = reinterpret_cast<T*>(_allocator.malloc(sizeof(T) * this->Tensor<T>::_shape.area()));
  if (_data == nullptr) {
    throw ResourceExhaustedError();
  }
  Device::getThreadDefault()->getInterface(T()).getDataInterface().set(
      _data, data.data(), this->Tensor<T>::_shape.area());
}

template <class T>
Constant<T>& Constant<T>::getConstant(const std::string& name, ProcFunc preprocessFunc) {
  Constant<T>& constant = Device::getThreadDefault()->getConstantManager(T()).getConstant(
      &Scope::getCurrentScope(), name);
  std::lock_guard<SpinLock> lck(constant._mtx);
  if (constant._data != nullptr) {
    return constant;
  }
  std::string fullName = Scope::getCurrentScope().getFullName() + name;
  std::string type, data;
  Shape shape;
  std::vector<T> realData;
  std::tie(shape, type, data) = DataLoader::getGlobalDefault()->loadData(fullName);
  realData = transfer_string_data<T>(type, data);
  if (realData.size() != shape.area()) {
    throw RuntimeError("DataLoader return a invalid data. Constant name: " + fullName);
  }
  if (preprocessFunc) {
    std::tie(shape, realData) = preprocessFunc(shape, realData);
  }
  constant.setData(shape, realData);
  return constant;
}

template <class T>
Constant<T>& Constant<T>::getConstantWithTranspose(
    const std::string& name, const std::vector<size_t>& perm) {
  return getConstant(name, [&perm](Shape shape, std::vector<T> data) {
    if (perm.size() != shape.size()) {
      throw InvalidArgumentError("perm and shape don't match");
    }
    std::vector<unsigned> newShape;
    for (size_t i = 0; i < perm.size(); ++i) {
      newShape.push_back(shape[perm[i]]);
    }
    return std::make_pair(Shape(newShape), host_transpose(data, shape, perm));
  });
}

template <class T>
Constant<T>* Constant<T>::tryGetRuntimeConstant(const std::string& name) {
  Constant<T>& constant = Device::getThreadDefault()->getConstantManager(T()).getConstant(
      &Scope::getCurrentScope(), "__runtime_constant_" + name);
  std::lock_guard<SpinLock> lck(constant._mtx);
  if (constant._data != nullptr) {
    return &constant;
  }
  return nullptr;
}

template <class T>
bool Constant<T>::setRuntimeConstantIfNotExist(
    const std::string& name, Shape shape, const std::vector<T>& data) {
  Constant<T>& constant = Device::getThreadDefault()->getConstantManager(T()).getConstant(
      &Scope::getCurrentScope(), "__runtime_constant_" + name);
  std::lock_guard<SpinLock> lck(constant._mtx);
  if (constant._data != nullptr) {
    return false;
  }
  constant.setData(shape, data);
  return true;
}

#define DEFINE_FUNC(type) template class Constant<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
