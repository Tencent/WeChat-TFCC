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

#include <limits>

#include "tfcc_mklreduceinterface.h"

#include "exceptions/tfcc_invalidargumenterror.h"
#include "exceptions/tfcc_notimplementederror.h"
#include "framework/tfcc_mklsession.h"
#include "framework/tfcc_session.h"
#include "framework/tfcc_types.h"
#include "interfaces/tfcc_mklinterfacehelper.h"

namespace tfcc {

template <class T>
void _mkl_reduce_sum(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    T reduceVal = static_cast<T>(0);
    const T* start = a + i * reduceSize;
    for (unsigned j = 0; j < reduceSize; ++j) {
      reduceVal += start[j];
    }
    b[i] = reduceVal;
  }
}

template <class T>
void _mkl_reduce_mean(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    T reduceVal = static_cast<T>(0);
    const T* start = a + i * reduceSize;
    for (unsigned j = 0; j < reduceSize; ++j) {
      reduceVal += start[j];
    }
    b[i] = reduceVal / static_cast<T>(reduceSize);
  }
}

template <class T>
void _mkl_reduce_prod(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    T reduceVal = static_cast<T>(1);
    const T* start = a + i * reduceSize;
    for (unsigned j = 0; j < reduceSize; ++j) {
      reduceVal *= start[j];
    }
    b[i] = reduceVal;
  }
}

template <class T>
void _mkl_reduce_max(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    const T* start = a + i * reduceSize;
    T reduceVal = start[0];
    for (unsigned j = 1; j < reduceSize; ++j) {
      reduceVal = std::max(start[j], reduceVal);
    }
    b[i] = reduceVal;
  }
}

template <class T>
void _mkl_reduce_min(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    const T* start = a + i * reduceSize;
    T reduceVal = start[0];
    for (unsigned j = 1; j < reduceSize; ++j) {
      reduceVal = std::min(start[j], reduceVal);
    }
    b[i] = reduceVal;
  }
}

template <class T>
void _mkl_reduce_any(const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
#pragma omp parallel for
  for (unsigned i = 0; i < chunkSize; ++i) {
    const T* start = a + i * reduceSize;
    T reduceVal = start[0];
    for (unsigned j = 1; j < reduceSize; ++j) {
      reduceVal = (start[j] || reduceVal) ? static_cast<T>(true) : static_cast<T>(false);
    }
    b[i] = reduceVal;
  }
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceSum(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_sum",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_sum(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceMean(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_mean",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_mean(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceProd(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_prod",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_prod(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceMax(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_max",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_max(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceMin(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_min",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_min(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

template <class T>
Variable<T> MKLReduceInterface<T>::reduceAny(const Tensor<T>& a, size_t keep) {
  std::vector<unsigned> s = a.shape().toVector();
  for (size_t i = keep; i < s.size(); ++i) {
    s[i] = 1;
  }
  Variable<T> result(s);
  unsigned reduceSize = 1;
  for (size_t i = keep; i < a.shape().size(); ++i) {
    reduceSize *= a.shape(i);
  }
  unsigned chunkSize = a.size() / reduceSize;

  mkl_async_wrapper(
      "reduce_any",
      [](const T* a, unsigned chunkSize, unsigned reduceSize, T* b) {
        _mkl_reduce_any(a, chunkSize, reduceSize, b);
      },
      a.data(), chunkSize, reduceSize, result.data());
  return result;
}

#define DEFINE_FUNC(type) template class MKLReduceInterface<type>;

TFCC_FOR_ALL_TYPES(DEFINE_FUNC);

}  // namespace tfcc
