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

#include "tfcc_embeddinglookuplayer.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_constant.h"
#include "framework/tfcc_scope.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "operations/tfcc_base.h"
#include "operations/tfcc_nn.h"

namespace tfcc {

namespace layer {

template <class T>
EmbeddingLookup<T>::EmbeddingLookup(std::string name)
    : EmbeddingLookup(std::move(name), EmbeddingPartitionType::NONE, 0) {}

template <class T>
EmbeddingLookup<T>::EmbeddingLookup(
    std::string name, EmbeddingPartitionType type, size_t partitionCount)
    : _initialized(false),
      _name(std::move(name)),
      _type(type),
      _partitionCount(partitionCount),
      _totalSize(0) {}

template <class T>
Variable<T> EmbeddingLookup<T>::operator()(const std::vector<unsigned>& ids) {
  if (!_initialized) {
    if (_type == EmbeddingPartitionType::NONE) {
      Constant<T>* c = &Constant<T>::getConstant(_name);
      _totalSize = c->shape(0);
      _embeddings.push_back(c);
    } else if (_type == EmbeddingPartitionType::MOD || _type == EmbeddingPartitionType::DIV) {
      if (_partitionCount == 0) {
        throw InvalidArgumentError("invalid partition count");
      }
      auto scopeG = Scope::scope(_name);
      for (size_t i = 0; i < _partitionCount; ++i) {
        Constant<T>* c = &Constant<T>::getConstant("part_" + std::to_string(i));
        _totalSize += c->shape(0);
        _embeddings.push_back(c);
      }
    } else {
      throw InvalidArgumentError("invalid partition type");
    }
  }

  if (_type == EmbeddingPartitionType::NONE) {
    return nn::embedding_lookup(*_embeddings[0], ids);
  }
  std::vector<View<T>> views;
  for (unsigned id : ids) {
    unsigned index, part;
    if (_type == EmbeddingPartitionType::MOD) {
      part = id % _partitionCount;
      index = id / _partitionCount;
    } else {
      unsigned size = (_totalSize + _partitionCount - 1) / _partitionCount;
      part = id / size;
      index = id % size;
    }
    Constant<T>* e = _embeddings[part];
    View<T> v = View<T>(*e, e->shape(), index, index + 1);
    views.push_back(v);
  }
  std::vector<const Tensor<T>*> tensors;
  for (const Tensor<T>& t : views) {
    tensors.push_back(&t);
  }
  return base::concat(tensors, 0);
}

template <class T>
Variable<T> EmbeddingLookup<T>::operator()(const std::vector<unsigned>& ids, const Shape& idShape) {
  if (idShape.area() != ids.size()) {
    throw InvalidArgumentError("ids.size() and idShape don't match");
  }

  Variable<T> result = this->operator()(ids);
  auto s = result.shape().toVector();
  s.erase(s.begin());
  auto idShapeVector = idShape.toVector();
  s.insert(s.begin(), idShapeVector.begin(), idShapeVector.end());
  result.reshape(s);
  return result;
}

#define DEFINE_FUNC(type) template class EmbeddingLookup<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace layer

}  // namespace tfcc
