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

#include "tfcc_math.h"

#include <algorithm>
#include <tuple>
#include <utility>

#include "exceptions/tfcc_invalidargumenterror.h"
#include "framework/tfcc_shape.h"
#include "framework/tfcc_types.h"
#include "framework/tfcc_view.h"
#include "interfaces/tfcc_activationinterface.h"
#include "interfaces/tfcc_arithmeticinterface.h"
#include "interfaces/tfcc_basicinterface.h"
#include "interfaces/tfcc_batcharithmeticinterface.h"
#include "interfaces/tfcc_minmaxinterface.h"
#include "interfaces/tfcc_reduceinterface.h"
#include "interfaces/tfcc_segmentinterface.h"
#include "interfaces/tfcc_transformationinterface.h"
#include "operations/tfcc_operation.h"

namespace tfcc {
namespace math {

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
Variable<T> add(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  if (resultS.area() == 0) {
    return tfcc::Variable<T>(resultS);
  }
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getArithmeticInterface().add(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> sub(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  if (resultS.area() == 0) {
    return tfcc::Variable<T>(resultS);
  }
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getArithmeticInterface().sub(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> mul(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  if (resultS.area() == 0) {
    return tfcc::Variable<T>(resultS);
  }
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getArithmeticInterface().mul(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> div(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  if (resultS.area() == 0) {
    return tfcc::Variable<T>(resultS);
  }
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getArithmeticInterface().div(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> batch_add(const std::vector<const Tensor<T>*>& values) {
  if (values.size() == 0) {
    throw InvalidArgumentError("values.size could not be zero");
  }
  auto shape = values[0]->shape();
  for (auto* t : values) {
    if (t == nullptr) {
      throw InvalidArgumentError("nullptr must not in values");
    }
    if (t->shape() != shape) {
      throw InvalidArgumentError("invalid shape");
    }
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBatchArithmeticInterface().add(values);
}

template <class T>
Variable<T> batch_add(std::initializer_list<const Tensor<T>*> values) {
  std::vector<const Tensor<T>*> vs(values);
  return batch_add(vs);
}

template <class T>
Variable<T> batch_mul(const std::vector<const Tensor<T>*>& values) {
  if (values.size() == 0) {
    throw InvalidArgumentError("values.size could not be zero");
  }
  auto shape = values[0]->shape();
  for (auto* t : values) {
    if (t == nullptr) {
      throw InvalidArgumentError("nullptr must not in values");
    }
    if (t->shape() != shape) {
      throw InvalidArgumentError("invalid shape");
    }
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBatchArithmeticInterface().mul(values);
}

template <class T>
Variable<T> batch_mul(std::initializer_list<const Tensor<T>*> values) {
  std::vector<const Tensor<T>*> vs(values);
  return batch_mul(vs);
}

template <class T>
Variable<T> transform(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform(a, alpha, beta);
}

template <class T>
Variable<T> transform2(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform2(a, alpha, beta);
}

template <class T>
Variable<T> transform3(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform3(a, alpha, beta);
}

template <class T>
Variable<T> transform4(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform4(a, alpha, beta);
}

template <class T>
Variable<T> transform5(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform5(a, alpha, beta);
}

template <class T>
Variable<T> transform6(const Tensor<T>& a, T alpha, T beta) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getTransformationInterface().transform6(a, alpha, beta);
}

template <class T>
Variable<T> min(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getMinMaxInterface().min(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> min(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getMinMaxInterface().min(a, b);
}

template <class T>
Variable<T> min(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getMinMaxInterface().min(b, a);
}

template <class T>
Variable<T> max(const Tensor<T>& a, const Tensor<T>& b) {
  Shape resultS, newAS, newBS;
  std::tie(resultS, newAS, newBS) = _get_broadcast_shape(a.shape(), b.shape());
  View<T> aView(a, std::move(newAS));
  View<T> bView(b, std::move(newBS));

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  tfcc::Variable<T> result = interface.getMinMaxInterface().max(aView, bView);
  result.reshape(std::move(resultS));
  return result;
}

template <class T>
Variable<T> max(const Tensor<T>& a, T b) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getMinMaxInterface().max(a, b);
}

template <class T>
Variable<T> max(T a, const Tensor<T>& b) {
  if (b.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getMinMaxInterface().max(b, a);
}

template <class T>
Variable<T> sigmoid(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().sigmoid(a);
}

template <class T>
Variable<T> relu(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().relu(a);
}

template <class T>
Variable<T> leaky_relu(const Tensor<T>& a, T alpha) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().leakyRelu(a, alpha);
}

template <class T>
Variable<T> softplus(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().softplus(a);
}

template <class T>
Variable<T> log(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().log(a);
}

template <class T>
Variable<T> rsqrt(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().rsqrt(a);
}

template <class T>
Variable<T> erf(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().erf(a);
}

template <class T>
Variable<T> tanh(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().tanh(a);
}

template <class T>
Variable<T> sin(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().sin(a);
}

template <class T>
Variable<T> cos(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().cos(a);
}

template <class T>
Variable<T> pow(const Tensor<T>& a, T exponent) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().pow(a, exponent);
}

template <class T>
Variable<T> pow(const Tensor<T>& a, const Tensor<T>& exponent) {
  if (a.size() == 0 || exponent.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  if (a.shape() == exponent.shape()) {
    return interface.getActivationInterface().pow(a, exponent);
  }
  Variable<T> rA = add(a, transform(exponent, static_cast<T>(0), static_cast<T>(0)));
  Variable<T> rExponent = add(exponent, transform(a, static_cast<T>(0), static_cast<T>(0)));
  return interface.getActivationInterface().pow(rA, rExponent);
}

template <class T>
Variable<T> pow(T a, const Tensor<T>& exponent) {
  if (exponent.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().pow(a, exponent);
}

template <class T>
Variable<T> softmax(const Tensor<T>& a, size_t axis) {
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().softmax(a, axis);
}

template <class T>
Variable<T> reduce_sum(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceSum(a, keep);
}

template <class T>
Variable<T> reduce_mean(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceMean(a, keep);
}

template <class T>
Variable<T> reduce_prod(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceProd(a, keep);
}

template <class T>
Variable<T> reduce_max(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceMax(a, keep);
}

template <class T>
Variable<T> reduce_min(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceMin(a, keep);
}

template <class T>
Variable<T> reduce_any(const Tensor<T>& a, size_t keep) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (keep < 0 || keep >= a.shape().size()) {
    throw InvalidArgumentError("invalid keep dims");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getReduceInterface().reduceAny(a, keep);
}

template <class T>
Variable<T> clip(const Tensor<T>& a, T minValue, T maxValue) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().clip(a, minValue, maxValue);
}

template <class T>
std::tuple<Variable<T>, Variable<uint32_t>> top_k(const Tensor<T>& a, unsigned k) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (a.shape(a.shape().size() - 1) < k) {
    throw InvalidArgumentError("invalid k");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().topK(a, k);
}

template <class T>
Variable<T> unsorted_segment_sum(const Tensor<T>& a, const Tensor<int>& ids, unsigned num) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (ids.shape().size() != 1) {
    throw InvalidArgumentError("invalid ids");
  }
  if (ids.size() != a.shape(0)) {
    throw InvalidArgumentError("tensor and ids don't match");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getSegmentInterface().unsortedSegmentSum(a, ids, num);
}

template <class T>
Variable<T> gelu(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().gelu(a);
}

template <class T>
Variable<T> gelu_accurate(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().geluAccurate(a);
}

template <class T>
Variable<T> abs(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().abs(a);
}

template <class T>
Variable<T> asin(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().asin(a);
}

template <class T>
Variable<T> asinh(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().asinh(a);
}

template <class T>
Variable<T> acos(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().acos(a);
}

template <class T>
Variable<T> acosh(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().acosh(a);
}

template <class T>
Variable<T> atan(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().atan(a);
}

template <class T>
Variable<T> atanh(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().atanh(a);
}

template <class T>
Variable<T> sign(const Tensor<T>& a) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }

  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getActivationInterface().sign(a);
}

template <class T>
Variable<int64_t> argmax(const Tensor<T>& a, size_t axis) {
  if (a.size() == 0) {
    throw InvalidArgumentError("invalid tensor");
  }
  if (axis >= a.shape().size()) {
    throw InvalidArgumentError("invalid axis");
  }
  Interface<T>& interface = Operation<T>::getCurrentInterface();
  return interface.getBasicInterface().argmax(a, axis);
}

#define DEFINE_FUNC(type)                                                               \
  template Variable<type> add(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> sub(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> mul(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> div(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> batch_add(const std::vector<const Tensor<type>*>& values);    \
  template Variable<type> batch_add(std::initializer_list<const Tensor<type>*> values); \
  template Variable<type> batch_mul(const std::vector<const Tensor<type>*>& values);    \
  template Variable<type> batch_mul(std::initializer_list<const Tensor<type>*> values); \
  template Variable<type> transform(const Tensor<type>& a, type alpha, type beta);      \
  template Variable<type> transform2(const Tensor<type>& a, type alpha, type beta);     \
  template Variable<type> transform3(const Tensor<type>& a, type alpha, type beta);     \
  template Variable<type> transform4(const Tensor<type>& a, type alpha, type beta);     \
  template Variable<type> transform5(const Tensor<type>& a, type alpha, type beta);     \
  template Variable<type> transform6(const Tensor<type>& a, type alpha, type beta);     \
  template Variable<type> min(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> min(const Tensor<type>& a, type b);                           \
  template Variable<type> min(type a, const Tensor<type>& b);                           \
  template Variable<type> max(const Tensor<type>& a, const Tensor<type>& b);            \
  template Variable<type> max(const Tensor<type>& a, type b);                           \
  template Variable<type> max(type a, const Tensor<type>& b);                           \
  template Variable<type> sigmoid(const Tensor<type>& a);                               \
  template Variable<type> relu(const Tensor<type>& a);                                  \
  template Variable<type> leaky_relu(const Tensor<type>& a, type alpha);                \
  template Variable<type> softplus(const Tensor<type>& a);                              \
  template Variable<type> log(const Tensor<type>& a);                                   \
  template Variable<type> rsqrt(const Tensor<type>& a);                                 \
  template Variable<type> erf(const Tensor<type>& a);                                   \
  template Variable<type> tanh(const Tensor<type>& a);                                  \
  template Variable<type> sin(const Tensor<type>& a);                                   \
  template Variable<type> cos(const Tensor<type>& a);                                   \
  template Variable<type> pow(const Tensor<type>& a, type exponent);                    \
  template Variable<type> pow(const Tensor<type>& a, const Tensor<type>& exponent);     \
  template Variable<type> pow(type a, const Tensor<type>& exponent);                    \
  template Variable<type> softmax(const Tensor<type>& a, size_t axis);                  \
  template Variable<type> reduce_sum(const Tensor<type>& a, size_t keep);               \
  template Variable<type> reduce_mean(const Tensor<type>& a, size_t keep);              \
  template Variable<type> reduce_prod(const Tensor<type>& a, size_t keep);              \
  template Variable<type> reduce_max(const Tensor<type>& a, size_t keep);               \
  template Variable<type> reduce_min(const Tensor<type>& a, size_t keep);               \
  template Variable<type> reduce_any(const Tensor<type>& a, size_t keep);               \
  template Variable<type> clip(const Tensor<type>& a, type minValue, type maxValue);    \
  template std::tuple<Variable<type>, Variable<uint32_t>> top_k(                        \
      const Tensor<type>& a, unsigned k);                                               \
  template Variable<type> unsorted_segment_sum(                                         \
      const Tensor<type>& a, const Tensor<int>& ids, unsigned num);                     \
  template Variable<type> gelu(const Tensor<type>& a);                                  \
  template Variable<type> gelu_accurate(const Tensor<type>& a);                         \
  template Variable<type> abs(const Tensor<type>& a);                                   \
  template Variable<type> asin(const Tensor<type>& a);                                  \
  template Variable<type> asinh(const Tensor<type>& a);                                 \
  template Variable<type> acos(const Tensor<type>& a);                                  \
  template Variable<type> acosh(const Tensor<type>& a);                                 \
  template Variable<type> atan(const Tensor<type>& a);                                  \
  template Variable<type> atanh(const Tensor<type>& a);                                 \
  template Variable<type> sign(const Tensor<type>& a);                                  \
  template Variable<int64_t> argmax(const Tensor<type>& a, size_t keep);

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

#define COMPLEX_DEFINE_FUNC(type) \
  template Variable<type> mul(const Tensor<type>& a, const Tensor<type>& b);

TFCC_FOR_COMPLEX_TYPES(COMPLEX_DEFINE_FUNC);

}  // namespace math
}  // namespace tfcc
