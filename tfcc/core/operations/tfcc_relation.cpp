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

#include "tfcc_relation.h"

#include <chrono>
#include <random>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_arithmeticinterface.h"
#include "interfaces/tfcc_comparisoninterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace relation {

// return make_tuple(result.shape, s1.shape, s2.shape)
static std::tuple<Shape, Shape, Shape> _get_broadcast_shape(const Shape& s1, const Shape& s2) {
  enum class MatchType { Match, BroadcastS1, BroadcastS2 };

  if (s1.size() == 0 || s2.size() == 0) {
    throw InvalidArgumentError("invalid tensor shape");
  }

  std::vector<unsigned> resultS, newS1, newS2;
  resultS = s1.size() > s2.size() ? s1.toVector() : s2.toVector();
  for (size_t i = 0; i < s1.size() && i < s2.size(); ++i) {
    unsigned l1 = s1[s1.size() - 1 - i];
    unsigned l2 = s2[s2.size() - 1 - i];
    if (l1 != l2 && l1 != 1 && l2 != 1) {
      throw InvalidArgumentError("broadcast failed");
    }
    resultS[resultS.size() - 1 - i] = std::max(l1, l2);
  }

  // squeeze
  size_t maxLen = std::max(s1.size(), s2.size());
  unsigned s1Len = s1[s1.size() - 1], s2Len = s2[s2.size() - 1];
  MatchType type = s1Len == s2Len ? MatchType::Match
                                  : s1Len == 1 ? MatchType::BroadcastS1 : MatchType::BroadcastS2;
  for (size_t i = 1; i < maxLen; ++i) {
    unsigned l1 = i < s1.size() ? s1[s1.size() - 1 - i] : 1;
    unsigned l2 = i < s2.size() ? s2[s2.size() - 1 - i] : 1;
    MatchType newType =
        l1 == l2 ? MatchType::Match : l1 == 1 ? MatchType::BroadcastS1 : MatchType::BroadcastS2;
    if (newType != type) {
      newS1.insert(newS1.begin(), s1Len);
      newS2.insert(newS2.begin(), s2Len);
      type = newType;
      s1Len = l1;
      s2Len = l2;
    } else {
      s1Len *= l1;
      s2Len *= l2;
    }
  }

  if (s1Len != 1 || s2Len != 1 || newS1.size() == 0 || newS2.size() == 0) {
    newS1.insert(newS1.begin(), s1Len);
    newS2.insert(newS2.begin(), s2Len);
  }

  return std::make_tuple(
      Shape(std::move(resultS)), Shape(std::move(newS1)), Shape(std::move(newS2)));
}

template <class T>
Variable<uint8_t> equal(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().equal(a, b);
}

template <class T>
Variable<uint8_t> equal(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().equal(b, a);
}

template <class T>
Variable<uint8_t> equal(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().equal(comp, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> unequal(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().unequal(a, b);
}

template <class T>
Variable<uint8_t> unequal(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().unequal(b, a);
}

template <class T>
Variable<uint8_t> unequal(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().unequal(comp, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> greater(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().greater(a, b);
}

template <class T>
Variable<uint8_t> greater(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().less(b, a);
}

template <class T>
Variable<uint8_t> greater(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().greater(comp, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> greater_equal(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().greaterEqual(a, b);
}

template <class T>
Variable<uint8_t> greater_equal(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().lessEqual(b, a);
}

template <class T>
Variable<uint8_t> greater_equal(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().greaterEqual(comp, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> less(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().less(a, b);
}

template <class T>
Variable<uint8_t> less(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().greater(b, a);
}

template <class T>
Variable<uint8_t> less(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().less(comp, static_cast<T>(0));
}

template <class T>
Variable<uint8_t> less_equal(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().lessEqual(a, b);
}

template <class T>
Variable<uint8_t> less_equal(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError();
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getComparisonInterface().greaterEqual(b, a);
}

template <class T>
Variable<uint8_t> less_equal(const Tensor<T>& a, const Tensor<T>& b) {
  if (a.size() == 0 || b.size() == 0) {
    throw InvalidArgumentError();
  }

  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  Variable<T> comp = interface.getArithmeticInterface().sub(aView, bView);
  comp.reshape(std::move(resultS));
  return interface.getComparisonInterface().lessEqual(comp, static_cast<T>(0));
}

#define DEFINE_FUNC(type)                                                                 \
  template Variable<uint8_t> equal(const Tensor<type>& a, type b);                        \
  template Variable<uint8_t> equal(type a, const Tensor<type>& b);                        \
  template Variable<uint8_t> equal(const Tensor<type>& a, const Tensor<type>& b);         \
  template Variable<uint8_t> unequal(const Tensor<type>& a, type b);                      \
  template Variable<uint8_t> unequal(type a, const Tensor<type>& b);                      \
  template Variable<uint8_t> unequal(const Tensor<type>& a, const Tensor<type>& b);       \
  template Variable<uint8_t> greater(const Tensor<type>& a, type b);                      \
  template Variable<uint8_t> greater(type a, const Tensor<type>& b);                      \
  template Variable<uint8_t> greater(const Tensor<type>& a, const Tensor<type>& b);       \
  template Variable<uint8_t> greater_equal(const Tensor<type>& a, type b);                \
  template Variable<uint8_t> greater_equal(type a, const Tensor<type>& b);                \
  template Variable<uint8_t> greater_equal(const Tensor<type>& a, const Tensor<type>& b); \
  template Variable<uint8_t> less(const Tensor<type>& a, type b);                         \
  template Variable<uint8_t> less(type a, const Tensor<type>& b);                         \
  template Variable<uint8_t> less(const Tensor<type>& a, const Tensor<type>& b);          \
  template Variable<uint8_t> less_equal(const Tensor<type>& a, type b);                   \
  template Variable<uint8_t> less_equal(type a, const Tensor<type>& b);                   \
  template Variable<uint8_t> less_equal(const Tensor<type>& a, const Tensor<type>& b);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace relation
}  // namespace tfcc
